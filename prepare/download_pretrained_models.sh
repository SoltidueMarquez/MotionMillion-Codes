mkdir -p checkpoints/pretrained_models
cd checkpoints/pretrained_models

echo "The pretrained models will be stored in the './checkpoints/pretrained_models' folder"

echo "Downloading"

# download the pretrained models trained on train.txt
# download 3B model
gdown --fuzzy https://drive.google.com/file/d/1IV3HDG8MdZz_Dd9xqyoHvh70zi6sodl6/view?usp=sharing
# download 7B model
gdown --fuzzy https://drive.google.com/file/d/1iwMDXEpATRRCxXGbXQewVax2HXpO9nOl/view?usp=sharing

# download the pretrained models trained on all.txt
# download 3B model
gdown --fuzzy https://drive.google.com/file/d/1wP_A4-fa213WqdkDPkaK34sKlJx1hBdP/view?usp=sharing
# download 7B model (The best performance model)
gdown --fuzzy https://drive.google.com/file/d/1kFdzFp_n1CfXChd2CFXjqJCL4TdEPbiJ/view?usp=sharing

echo "Downloading done!"
