CUDA_VISIBLE_DEVICES=0 \
python inference.py \
--det_model ./weights/model_yolo.pt \
--rec_model ./weights/model_crnn.pth \
--data_path /workspace/LPDR/Database/LPR/test \
--save_path ./result/plr_val
