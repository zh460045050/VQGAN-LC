
####Extract path-level features of training images
CUDA_VISIBLE_DEVICES=0 python clip_feature_generation.py --batch_size 4096 --imagenet_path $imagenet_path

####Cluster the features to generate initialized codebook
CUDA_VISIBLE_DEVICES=0 python minibatch_kmeans_per_class.py --start 0 \
                                                            --end 1000 \
                                                            --n_class 1000 \
                                                            --downsample 4 \
                                                            --save_dir "clustering_centers_1000" \
                                                            --imagenet_feature_path "Imagenet_clip_features/train"