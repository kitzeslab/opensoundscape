#!/usr/bin/env python3
"""
Augmentations and transforms for torch.Tensors

These functions were implemented for PyTorch in:
https://github.com/zcaceres/spec_augment
The original paper is available on https://arxiv.org/abs/1904.08779
"""

import random
import torch


def time_warp(spec, W=5):
    """apply time stretch and shearing to spectrogram

    fills empty space on right side with horizontal bars

    W controls amount of warping. Random with occasional large warp.
    """
    batch_size = spec.shape[0]
    num_channel = spec.shape[1]
    num_rows = spec.shape[2]
    spec_len = spec.shape[3]
    device = spec.device

    if num_channel != 1:
        raise NameError("Warping only supports one channel input currently.")

    spec = spec.reshape(batch_size, num_rows, spec_len)

    y = num_rows // 2

    horizontal_line_at_ctr = spec[:, y]

    point_to_warp = horizontal_line_at_ctr[
        range(batch_size),
        [random.randrange(W, spec_len - W) for _ in range(batch_size)],
    ]

    # Uniform distribution from (0,W) with chance to be up to W negative
    dist_to_warp = [random.randrange(-W, W) for _ in range(batch_size)]

    src_pts = torch.tensor(
        [[[y, point_to_warp[i]]] for i in range(batch_size)], device=device
    )
    dest_pts = torch.tensor(
        [[[y, point_to_warp[i] + dist_to_warp[i]]] for i in range(batch_size)],
        device=device,
    )

    warped_spectro, dense_flows = sparse_image_warp(spec, src_pts, dest_pts)
    return warped_spectro.squeeze(3).reshape(
        batch_size, num_channel, num_rows, spec_len
    )


def sparse_image_warp(
    img_tensor,
    source_control_point_locations,
    dest_control_point_locations,
    interpolation_order=2,
    regularization_weight=0.0,
    num_boundaries_points=0,
):

    device = img_tensor.device
    control_point_flows = dest_control_point_locations - source_control_point_locations

    batch_size, image_height, image_width = img_tensor.shape
    flattened_grid_locations = torch.cat(
        [get_flat_grid_locations(image_height, image_width, device).unsqueeze(0)]
        * batch_size,
        dim=0,
    )

    flattened_flows = interpolate_spline(
        dest_control_point_locations,
        control_point_flows,
        flattened_grid_locations,
        interpolation_order,
        regularization_weight,
    )

    dense_flows = create_dense_flows(
        flattened_flows, batch_size, image_height, image_width
    )

    warped_image = dense_image_warp(img_tensor, dense_flows)

    return warped_image, dense_flows


def get_flat_grid_locations(image_height, image_width, device):
    y_range = torch.linspace(0, image_height - 1, image_height, device=device)
    x_range = torch.linspace(0, image_width - 1, image_width, device=device)
    y_grid, x_grid = torch.meshgrid(y_range, x_range)
    return torch.stack((y_grid, x_grid), -1).reshape([image_height * image_width, 2])


def interpolate_spline(
    train_points, train_values, query_points, order, regularization_weight=0.0
):
    # First, fit the spline to the observed data.
    w, v = solve_interpolation(train_points, train_values, order, regularization_weight)
    # Then, evaluate the spline at the query locations.
    query_values = apply_interpolation(query_points, train_points, w, v, order)

    return query_values


def solve_interpolation(train_points, train_values, order, regularization_weight):

    device = train_points.device
    b, n, d = train_points.shape
    k = train_values.shape[-1]

    c = train_points
    f = train_values.float()

    matrix_a = phi(cross_squared_distance_matrix(c, c), order).view(
        b, n, n
    )  # [b, n, n]

    # Append ones to the feature values for the bias term in the linear model.
    ones = torch.ones((b, 1, 1), dtype=train_points.dtype, device=device)
    matrix_b = torch.cat((c, ones), 2).float()  # [b, n, d + 1]

    # [b, n + d + 1, n]
    left_block = torch.cat((matrix_a, torch.transpose(matrix_b, 2, 1)), dim=1)

    num_b_cols = matrix_b.shape[2]  # d + 1

    # In Tensorflow, zeros are used here. Pytorch solve fails with zeros for some reason we don't understand.
    # So instead we use very tiny randn values (variance of one, zero mean) on one side of our multiplication.
    lhs_zeros = torch.randn((b, num_b_cols, num_b_cols), device=device) / 1e10
    right_block = torch.cat((matrix_b, lhs_zeros), dim=1)  # [b, n + d + 1, d + 1]
    lhs = torch.cat((left_block, right_block), dim=2)  # [b, n + d + 1, n + d + 1]

    rhs_zeros = torch.zeros(
        (b, d + 1, k), dtype=train_points.dtype, device=device
    ).float()
    rhs = torch.cat((f, rhs_zeros), 1)  # [b, n + d + 1, k]

    # Then, solve the linear system and unpack the results.
    X, LU = torch.solve(rhs, lhs)
    w = X[:, :n, :]
    v = X[:, n:, :]

    return w, v


def cross_squared_distance_matrix(x, y):
    return (x - y).pow(2).sum(dim=-1)


def phi(r, order):
    EPSILON = torch.tensor(1e-10, device=r.device)
    # using EPSILON prevents log(0), sqrt0), etc.
    if order == 1:
        r = torch.max(r, EPSILON)
        r = torch.sqrt(r)
        return r
    elif order == 2:
        return 0.5 * r * torch.log(torch.max(r, EPSILON))
    elif order == 4:
        return 0.5 * torch.square(r) * torch.log(torch.max(r, EPSILON))
    elif order % 2 == 0:
        r = torch.max(r, EPSILON)
        return 0.5 * torch.pow(r, 0.5 * order) * torch.log(r)
    else:
        r = torch.max(r, EPSILON)
        return torch.pow(r, 0.5 * order)


def apply_interpolation(query_points, train_points, w, v, order):

    # First, compute the contribution from the rbf term.

    pairwise_dists = cross_squared_distance_matrix(
        query_points.float(), train_points.float()
    ).reshape(query_points.shape[0], query_points.shape[1], -1)
    phi_pairwise_dists = phi(pairwise_dists, order)

    rbf_term = torch.bmm(phi_pairwise_dists, w)

    # Then, compute the contribution from the linear term.
    # Pad query_points with ones, for the bias term in the linear model.
    ones = torch.ones_like(query_points[..., :1])
    query_points_pad = torch.cat((query_points, ones), 2).float()
    linear_term = torch.bmm(query_points_pad, v)

    return rbf_term + linear_term


def create_dense_flows(flattened_flows, batch_size, image_height, image_width):
    # possibly .view
    return torch.reshape(flattened_flows, [batch_size, image_height, image_width, 2])


def dense_image_warp(image, flow):

    image = image.unsqueeze(3)
    batch_size, height, width, channels = image.shape
    device = image.device

    # The flow is defined on the image grid. Turn the flow into a list of query
    # points in the grid space.
    grid_x, grid_y = torch.meshgrid(
        torch.arange(width, device=device), torch.arange(height, device=device)
    )

    stacked_grid = torch.stack((grid_y, grid_x), dim=2).float()

    batched_grid = torch.cat([stacked_grid.unsqueeze(0)] * batch_size, dim=0).permute(
        0, 2, 1, 3
    )

    query_points_on_grid = batched_grid - flow
    query_points_flattened = torch.reshape(
        query_points_on_grid, [batch_size, height * width, 2]
    )

    # Compute values at the query points, then reshape the result back to the
    # image grid.
    interpolated = interpolate_bilinear(image, query_points_flattened)
    interpolated = torch.reshape(interpolated, [batch_size, height, width, channels])
    return interpolated


def interpolate_bilinear(
    grid, query_points, name="interpolate_bilinear", indexing="ij"
):

    if indexing != "ij" and indexing != "xy":
        raise ValueError("Indexing mode must be 'ij' or 'xy'")

    shape = grid.shape
    if len(shape) != 4:
        msg = "Grid must be 4 dimensional. Received size: "
        raise ValueError(msg + str(grid.shape))

    batch_size, height, width, channels = grid.shape

    shape = [batch_size, height, width, channels]
    query_type = query_points.dtype
    grid_type = grid.dtype
    grid_device = grid.device

    num_queries = query_points.shape[1]

    alphas = []
    floors = []
    ceils = []
    index_order = [0, 1] if indexing == "ij" else [1, 0]
    unstacked_query_points = query_points.unbind(2)

    for dim in index_order:
        queries = unstacked_query_points[dim]

        size_in_indexing_dimension = shape[dim + 1]

        # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
        # is still a valid index into the grid.
        max_floor = torch.tensor(
            size_in_indexing_dimension - 2, dtype=query_type, device=grid_device
        )
        min_floor = torch.tensor(0.0, dtype=query_type, device=grid_device)
        maxx = torch.max(min_floor, torch.floor(queries))
        floor = torch.min(maxx, max_floor)
        int_floor = floor.long()
        floors.append(int_floor)
        ceil = int_floor + 1
        ceils.append(ceil)

        # alpha has the same type as the grid, as we will directly use alpha
        # when taking linear combinations of pixel values from the image.

        alpha = (queries - floor).float()
        min_alpha = torch.tensor(0.0, dtype=grid_type, device=grid_device)
        max_alpha = torch.tensor(1.0, dtype=grid_type, device=grid_device)
        alpha = torch.min(torch.max(min_alpha, alpha), max_alpha)

        # Expand alpha to [b, n, 1] so we can use broadcasting
        # (since the alpha values don't depend on the channel).
        alpha = torch.unsqueeze(alpha, 2)
        alphas.append(alpha)

    flattened_grid = torch.reshape(grid, [batch_size, height * width, channels])

    # This wraps array_ops.gather. We reshape the image data such that the
    # batch, y, and x coordinates are pulled into the first dimension.
    # Then we gather. Finally, we reshape the output back. It's possible this
    # code would be made simpler by using array_ops.gather_nd.

    def gather(y_coords, x_coords, name):
        linear_coordinates = (y_coords * width + x_coords).reshape(
            batch_size, height * width, channels
        )
        gathered_values = torch.gather(flattened_grid, 1, linear_coordinates)
        return gathered_values

    # grab the pixel values in the 4 corners around each query point
    top_left = gather(floors[0], floors[1], "top_left")
    top_right = gather(floors[0], ceils[1], "top_right")
    bottom_left = gather(ceils[0], floors[1], "bottom_left")
    bottom_right = gather(ceils[0], ceils[1], "bottom_right")

    interp_top = alphas[1] * (top_right - top_left) + top_left
    interp_bottom = alphas[1] * (bottom_right - bottom_left) + bottom_left
    interp = alphas[0] * (interp_bottom - interp_top) + interp_top

    return interp


def freq_mask(spec, F=30, max_masks=3, replace_with_zero=False):
    """draws horizontal bars over the image

    F:maximum frequency-width of bars in pixels

    max_masks: maximum number of bars to draw

    replace_with_zero: if True, bars are 0s, otherwise, mean img value
    """
    cloned = spec.clone()
    batch_size = cloned.shape[0]
    num_mel_channels = cloned.shape[2]

    num_masks = random.randint(1, max_masks)

    for _ in range(num_masks):

        f = [random.randrange(0, F) for _ in range(batch_size)]
        f_zero = [
            random.randrange(0, num_mel_channels - f[i])
            for i, _ in enumerate(range(batch_size))
        ]

        mask_end = [
            random.randrange(f_zero[i], f_zero[i] + f[i])
            if f_zero[i] != (f_zero[i] + f[i])
            else (f_zero[i] + F)
            for i, _ in enumerate(range(batch_size))
        ]

        if replace_with_zero:
            mask_value = [0.0] * batch_size
        else:
            mask_value = cloned.mean(dim=(1, 2, 3))

        for i in range(len(cloned)):
            cloned[i, :, f_zero[i] : mask_end[i], :] = mask_value[i]

    return cloned


def time_mask(spec, T=40, max_masks=3, replace_with_zero=False):
    """draws vertical bars over the image

    T:maximum time-width of bars in pixels

    max_masks: maximum number of bars to draw

    replace_with_zero: if True, bars are 0s, otherwise, mean img value
    """
    cloned = spec.clone()
    batch_size = cloned.shape[0]
    len_spectro = cloned.shape[3]

    num_masks = random.randint(1, max_masks)

    for _ in range(num_masks):

        t = [random.randrange(0, T) for _ in range(batch_size)]
        t_zero = [
            random.randrange(0, len_spectro - t[i])
            for i, _ in enumerate(range(batch_size))
        ]

        mask_end = [
            random.randrange(t_zero[i], t_zero[i] + t[i])
            if t_zero[i] != (t_zero[i] + t[i])
            else (t_zero[i] + T)
            for i, _ in enumerate(range(batch_size))
        ]

        if replace_with_zero:
            mask_value = [0.0] * batch_size
        else:
            mask_value = cloned.mean(dim=(1, 2, 3))

        for i in range(len(cloned)):
            cloned[i, :, :, t_zero[i] : mask_end[i]] = mask_value[i]

    return cloned
