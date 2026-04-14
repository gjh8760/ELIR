import torch
import math
import numpy as np


def create_plateau_blending_mask(patch_height, patch_width, overlap_size_y, overlap_size_x, min_weight):
    """
    Sigmoid-based 2D plateau blending mask: center=1, overlap edges smoothly ramp down.
    Returns a torch float tensor of shape (1, 1, patch_height, patch_width).
    """
    ramp_size_y = min(overlap_size_y, patch_height // 2)
    ramp_size_x = min(overlap_size_x, patch_width // 2)

    y = np.linspace(-6, 6, ramp_size_y)
    ramp_1d_y = 1 / (1 + np.exp(-y))
    x = np.linspace(-6, 6, ramp_size_x)
    ramp_1d_x = 1 / (1 + np.exp(-x))

    center_h = patch_height - 2 * ramp_size_y
    center_w = patch_width - 2 * ramp_size_x

    mask_y = np.concatenate([ramp_1d_y, np.ones(center_h), ramp_1d_y[::-1]])
    mask_x = np.concatenate([ramp_1d_x, np.ones(center_w), ramp_1d_x[::-1]])

    blending_mask = np.outer(mask_y, mask_x)
    scaled_mask = min_weight + blending_mask * (1 - min_weight)
    scaled_mask = torch.from_numpy(scaled_mask).float()
    return scaled_mask.unsqueeze(0).unsqueeze(0)


def get_model_size(model):
    count_params = sum([p.numel() for p in model.parameters()])
    return count_params


def cosin_metric(x1, x2):
    rad = torch.dot(x1, x2) / (torch.norm(x1) * torch.norm(x2))
    return torch.arccos(rad) / math.pi * 180


def minmax_norm(x, axis=[1,2,3]):
    return (x - x.amin(dim=axis, keepdim=True)) / (x.amax(dim=axis, keepdim=True) - x.amin(dim=axis, keepdim=True))


def rgb2ycbcr(img, y_only=True):
    """Convert RGB images to YCbCr images"""
    if y_only:
        weight = torch.tensor([[65.481], [128.553], [24.966]]).to(img)
        ycbcr = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = torch.tensor([[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]).to(img)
        bias = torch.tensor([16, 128, 128]).view(1, 3, 1, 1).to(img)
        ycbcr = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias
    ycbcr = ycbcr / 255.
    return ycbcr


def ycbcr2rgb(img):
    """Convert YcbCr images to RGB images"""
    weight = torch.tensor([[1.164, 1.164, 1.164], [0, -0.391, 2.017], [1.593, -0.8109, 0]]).to(img)
    bias = torch.tensor([-0.8742, 0.5316, -1.0856]).view(1, 3, 1, 1).to(img)
    rgb = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias
    return rgb


class ImageSpliterTh:
    def __init__(self, im, pch_size, stride, sf=1, extra_bs=1):
        '''
        Input:
            im: n x c x h x w, torch tensor, float, low-resolution image in SR
            pch_size, stride: patch setting. Each can be an int (applied to
                both H and W) or a (h, w) tuple/list for anisotropic patches.
            sf: scale factor in image super-resolution
            pch_bs: aggregate pchs to processing, only used when inputing single image
        '''
        pch_h, pch_w = self._pair(pch_size)
        stride_h, stride_w = self._pair(stride)
        assert stride_h <= pch_h and stride_w <= pch_w
        self.pch_h, self.pch_w = pch_h, pch_w
        self.stride_h, self.stride_w = stride_h, stride_w
        # Backward compatible scalar aliases (used only when h==w)
        self.pch_size = pch_h
        self.stride = stride_h
        self.sf = sf
        self.extra_bs = extra_bs

        bs, chn, height, width= im.shape
        self.true_bs = bs

        self.height_starts_list = self.extract_starts(height, pch_h, stride_h)
        self.width_starts_list = self.extract_starts(width, pch_w, stride_w)
        self.starts_list = []
        for ii in self.height_starts_list:
            for jj in self.width_starts_list:
                self.starts_list.append([ii, jj])

        self.length = self.__len__()
        self.count_pchs = 0

        self.im_ori = im
        self.im_res = torch.zeros([bs, chn, height*sf, width*sf], dtype=im.dtype, device=im.device)
        self.pixel_count = torch.zeros([bs, chn, height*sf, width*sf], dtype=im.dtype, device=im.device)

        out_pch_h = pch_h * sf
        out_pch_w = pch_w * sf
        overlap_y = max(pch_h - stride_h, 0) * sf
        overlap_x = max(pch_w - stride_w, 0) * sf
        self.blend_mask = create_plateau_blending_mask(
            patch_height=out_pch_h, patch_width=out_pch_w,
            overlap_size_y=overlap_y, overlap_size_x=overlap_x,
            min_weight=1e-6,
        ).to(device=im.device, dtype=im.dtype)

    @staticmethod
    def _pair(v):
        if isinstance(v, (list, tuple)):
            assert len(v) == 2, "pch_size/stride tuple must be (h, w)"
            return int(v[0]), int(v[1])
        return int(v), int(v)

    def extract_starts(self, length, pch, stride):
        if length <= pch:
            starts = [0,]
        else:
            starts = list(range(0, length, stride))
            for ii in range(len(starts)):
                if starts[ii] + pch > length:
                    starts[ii] = length - pch
            starts = sorted(set(starts), key=starts.index)
        return starts

    def __len__(self):
        return len(self.height_starts_list) * len(self.width_starts_list)

    def __iter__(self):
        return self

    def __next__(self):
        if self.count_pchs < self.length:
            index_infos = []
            current_starts_list = self.starts_list[self.count_pchs:self.count_pchs+self.extra_bs]
            for ii, (h_start, w_start) in enumerate(current_starts_list):
                w_end = w_start + self.pch_w
                h_end = h_start + self.pch_h
                current_pch = self.im_ori[:, :, h_start:h_end, w_start:w_end]
                if ii == 0:
                    pch = current_pch
                else:
                    pch = torch.cat([pch, current_pch], dim=0)

                h_start *= self.sf
                h_end *= self.sf
                w_start *= self.sf
                w_end *= self.sf
                index_infos.append([h_start, h_end, w_start, w_end])

            self.count_pchs += len(current_starts_list)
        else:
            raise StopIteration()

        return pch, index_infos

    def update(self, pch_res, index_infos):
        '''
        Input:
            pch_res: (n*extra_bs) x c x pch_size x pch_size, float
            index_infos: [(h_start, h_end, w_start, w_end),]
        '''
        assert pch_res.shape[0] % self.true_bs == 0
        pch_list = torch.split(pch_res, self.true_bs, dim=0)
        assert len(pch_list) == len(index_infos)
        for ii, (h_start, h_end, w_start, w_end) in enumerate(index_infos):
            current_pch = pch_list[ii]
            h_span = h_end - h_start
            w_span = w_end - w_start
            mask = self.blend_mask[:, :, :h_span, :w_span]
            self.im_res[:, :, h_start:h_end, w_start:w_end] += current_pch * mask
            self.pixel_count[:, :, h_start:h_end, w_start:w_end] += mask

    def gather(self):
        return self.im_res.div(self.pixel_count.clamp(min=1e-6))
