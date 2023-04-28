# exp_name = f"{dataset_filepath.split('.')[0]}_resnet18_sc1_iwu_fast"  # f"bbs25_iter2_resnet18_{n_features}_kmeans_{n_clusters}_haarlike4sc3_fast"

# for n_feature in n_features:
#     for n_cluster in n_clusters:
#         print(f"n_features: {n_feature}, n_clusters: {n_cluster}")
#         miou, sr = main(dataset_filepath, n_cluster, n_feature)
#         all_mious.append(miou)
#         all_success_rates.append(sr)
#         all_features.append(n_feature)
#         all_clusters.append(n_cluster)

# df.pivot("n_features", "n_clusters", "M_IOU").to_csv(
#     os.path.join(exp_folder, f"{exp_name}_mious.csv")
# )
# df.pivot("n_features", "n_clusters", "Success_Rate").to_csv(
#     os.path.join(exp_folder, f"{exp_name}_success_rates.csv")
# )

# print(f"Mean IOU: {np.mean(ious)}")
# ious1.append(bbox_iou_1.item())
# print(f"{i} {bbox_iou.item()} {bbox_iou_1.item()}")
# if i // 100 == 0:
#     torch.cuda.empty_cache()

# print(f"Mean IOU1: {np.mean(ious1)}")
# cv2.rectangle(
#     template_image,
#     (template_bbox[0], template_bbox[1]),
#     (template_bbox[0] + template_bbox[2], template_bbox[1] + template_bbox[3]),
#     (0, 255, 0),
#     2,
# )
# cv2.imwrite(
#     os.path.join(exp_folder, exp_name, f"{i}_template.png"),
#     cv2.cvtColor(template_image, cv2.COLOR_BGR2RGB),
# )

# query_x = query_x - template_bbox[3] // 2
# query_y = query_y - template_bbox[2] // 2

# cv2.rectangle(
#     query_image, (query_y, query_x), (query_y + query_h, query_x + query_w), (0, 255, 0), 3,
# )
# cv2.rectangle(
#     query_image,
#     (query_bbox[0], query_bbox[1]),
#     (query_bbox[0] + query_bbox[2], query_bbox[1] + query_bbox[3]),
#     (255, 0, 0),
#     3,
# )

# heatmap = heatmap / heatmap.max()
# heatmap = heatmap * 255
# heatmap = heatmap.astype(np.uint8)
# cv2.imwrite(os.path.join(exp_folder, exp_name, f"{i}_heatmap.png"), heatmap)

# heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
# heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
# cv2.imwrite(
#     os.path.join(exp_folder, exp_name, f"{i}_query.png"),
#     np.concatenate([cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB), heatmap], axis=1),
# )

# bbox_iou_1 = bops.box_iou(
#     torch.tensor(
#         [query_x, query_y, query_x + query_bbox[3], query_y + query_bbox[2]]
#     ).unsqueeze(0),
#     torch.tensor(
#         [
#             query_bbox[1],
#             query_bbox[0],
#             query_bbox[1] + query_bbox[3],
#             query_bbox[0] + query_bbox[2],
#         ]
#     ).unsqueeze(0),
# )


# cv2.imwrite("query_image.png", query_image)

# fig, ax = plt.subplots(nrows=1, ncols=2)  # , figsize=(16, 12))
# ax[0].imshow(template_image)
# ax[1].imshow(query_image)
# ax[1].imshow(heatmap, cmap="jet", alpha=0.6)

# ax[1].scatter(min_loc[:, 1], min_loc[:, 0], c="r", s=10)
# x = 1


# dataset_folder = "/mnt/hdd1/users/aktgpt/datasets/template_matching/TinyTLP/Aquarium1/img"
# imgs_path = sorted(glob.glob(os.path.join(dataset_folder, "*.jpg")))
# annotations = pd.read_csv(
#     "/mnt/hdd1/users/aktgpt/datasets/template_matching/TinyTLP/Aquarium1/groundtruth_rect.txt",
#     header=None,
#     index_col=False,
#     # sep="\t",
# )
# annotations.head()

# image = cv2.cvtColor(cv2.imread(imgs_path[0]), cv2.COLOR_BGR2RGB)
# # print(image.max(), image.min())

# # print(image.max(), image.min())
# feature_extractor = PixelFeatureExtractor(model_name="resnet18", num_features=64)
# device = feature_extractor.device

# images_features = feature_extractor.get_features(image)
# # print(images_features.shape, image.shape)
# # x, y, w, h = annotations.iloc[0, :].values
# x, y, w, h = annotations.iloc[0, 1:5].values

# patch = image[y : y + h, x : x + w, :]
# patch_features = images_features[:, y : y + h, x : x + w]

# template_matcher = TemplateMatcher(
#     patch_features, patch, images_features, [x, y, w, h], n_clusters=n_clusters
# )
# for i in range(1, len(imgs_path) // 2, 50):
#     # for img_path in imgs_path[85:95]:
#     img = cv2.cvtColor(cv2.imread(imgs_path[i]), cv2.COLOR_BGR2RGB)
#     image_features = feature_extractor.get_features(img)
#     # print(image_features.shape)

#     heatmap1, heatmap2 = template_matcher.get_heatmap(image_features)
#     # x, y = torch.where(heatmap1 == heatmap1.min())
#     fig, ax = plt.subplots(nrows=1, ncols=3)  # , figsize=(16, 12))
#     ax[0].imshow(img)
#     ax[1].imshow(img)
#     ax[1].imshow(heatmap1.cpu().numpy(), cmap="jet", alpha=0.6)
#     _, min_loc1 = torch.topk(-heatmap1.flatten(), 10)
#     min_loc1 = np.array(np.unravel_index(min_loc1.cpu().numpy(), heatmap1.shape)).T
#     ax[1].scatter(min_loc1[:, 1], min_loc1[:, 0], c="r", s=10)
#     # rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor="r", facecolor="none")
#     # ax.add_patch(rect)
#     # rect_det = patches.Rectangle(
#     #     (det_x[0].cpu().numpy(), det_y[0].cpu().numpy()),
#     #     w,
#     #     h,
#     #     linewidth=1,
#     #     edgecolor="g",
#     #     facecolor="none",
#     # )
#     # ax.add_patch(rect_det)
#     ax[2].imshow(img)
#     ax[2].imshow(heatmap2.cpu().numpy(), cmap="jet", alpha=0.6)
#     _, min_loc2 = torch.topk(-heatmap2.flatten(), 10)
#     min_loc2 = np.array(np.unravel_index(min_loc2.cpu().numpy(), heatmap2.shape)).T
#     ax[2].scatter(min_loc2[:, 1], min_loc2[:, 0], c="r", s=10)
# plt.show()

# c, w, h = patch_features.shape
# x = patch_features.reshape(c, w * h).transpose(1, 0)
# # print(x.shape)
# cluster_centers, choice_cluster, dis_vals = kmeans(X=x, num_clusters=n_clusters)
# print(dis_vals.min(), dis_vals.max())
# print(choice_cluster.shape, cluster_centers.shape)
# _, counts = torch.unique(choice_cluster, return_counts=True)
# feature_histogram = counts  # / (w * h)
# feature_norm = cluster_centers.norm(dim=1, keepdim=True)

# dist_calculator = nn.Conv1d(in_channels=c, out_channels=n_clusters, kernel_size=1, bias=False).to(
#     feature_extractor.device
# )
# dist_calculator.weight.data = cluster_centers.unsqueeze(-1).to(torch.float32)
# dist_calculator.weight.requires_grad = False

# pool1d = nn.MaxPool1d(kernel_size=n_clusters, return_indices=True)
# unpool1d = nn.MaxUnpool1d(kernel_size=n_clusters)

# sum_kern = torch.nn.functional.one_hot(choice_cluster.reshape(w, h)).permute(2, 0, 1).unsqueeze(1)
# sum_kern1 = dilation(sum_kern, torch.ones(5, 5).to(device))
# # fig, ax = plt.subplots(nrows=8, ncols=8)
# # i=0
# # for row in ax:
# #     for col in row:
# #         col.imshow(sum_kern1[i,0,:,:].cpu().numpy())
# #         i +=1
# sum_dis_filter = nn.Conv2d(
#     n_clusters,
#     n_clusters,
#     (w, h),
#     padding=(w // 2 + 1, h // 2 + 1),
#     padding_mode="reflect",
#     groups=n_clusters,
#     bias=False,
# ).to(device)
# # sum_dis_filter.weight.data = torch.ones(n_clusters, 1, w, h).to(device).to(torch.float32) #/ (w * h)
# sum_dis_filter.weight.data = sum_kern1
# # / sum_kern1.sum(dim=(1, 2, 3)).unsqueeze(-1).unsqueeze(
# # -1
# # ).unsqueeze(-1)
# sum_dis_filter.weight.requires_grad = False


# rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor="r", facecolor="none")
# ax.add_patch(rect)
# rect_det = patches.Rectangle(
#     (det_x[0].cpu().numpy(), det_y[0].cpu().numpy()),
#     w,
#     h,
#     linewidth=1,
#     edgecolor="g",
#     facecolor="none",
# )
# ax.add_patch(rect_det)
