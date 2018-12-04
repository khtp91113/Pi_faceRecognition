export PYTHONPATH=~/facenet/src/
python facenet/src/align/align_dataset_mtcnn.py --image_size 160 image/ image_align/
python create_face_embeddings.py
