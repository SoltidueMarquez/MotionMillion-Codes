mkdir -p checkpoints/
cd checkpoints/
git lfs install
git clone https://huggingface.co/google/flan-t5-xl

echo "The T5-XL will be stored in the './checkpoints' folder, named as models--google--flan-t5-xl "

cd ..