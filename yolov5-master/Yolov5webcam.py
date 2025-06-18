import cv2
import torch
from models.common import DetectMultiBackend
from utils.dataloaders import LoadStreams
from utils.general import non_max_suppression, scale_boxes
from utils.plots import Annotator, colors

def run_camera_detection(
    model_path="yolov5s.pt",
    camera_index=0,
    img_size=640,
    conf_threshold=0.25,
    iou_threshold=0.45,
    device="cpu"
):
    # Cihaz seçimi
    device = torch.device(device)
    
    # Modeli yükle
    model = DetectMultiBackend(model_path, device=device)
    model.eval()
    
    # Kamera akışını başlat
    dataset = LoadStreams(str(camera_index), img_size=img_size, auto=model.pt)
    # Kamera akışını işleme
    for path, img, im0s, _, _ in dataset:
        # Görüntüyü modele uygun hale getirme
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        if len(img.shape) == 3:
            img = img[None]  # Batch boyutu ekle

        # Tahmin yapma
        predictions = model(img)
        
        # Non-Maximum Suppression uygulama
        predictions = non_max_suppression(predictions, conf_threshold, iou_threshold)
        
        # Sonuçları işleme
        for i, det in enumerate(predictions):
            frame = im0s[i].copy()
            
            if len(det):
                # Kutuları orijinal görüntü boyutuna ölçekleme
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                
                # Görselleştirme
                annotator = Annotator(frame, line_width=2)
                for *xyxy, conf, cls in det:
                    # Kutu ve etiket ekleme
                    label = f"{model.names[int(cls)]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(int(cls), True))
            
            # Sonucu göster
            cv2.imshow("Camera Object Detection", frame)
            if cv2.waitKey(1) == ord('q'):  # 'q' tuşu ile çıkış
                return

if __name__ == "__main__":
    # Basit argüman işleme
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov5s.pt", help="Model dosya yolu")
    parser.add_argument("--camera", type=int, default=0, help="Kamera indeksi (0, 1, vb.)")
    parser.add_argument("--img-size", type=int, default=640, help="İşlem boyutu")
    parser.add_argument("--conf", type=float, default=0.25, help="Güven eşiği")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU eşiği")
    parser.add_argument("--device", default="cpu", help="cpu veya cuda")
    args = parser.parse_args()
    
    run_camera_detection(
        model_path=args.model,
        camera_index=args.camera,
        img_size=args.img_size,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device 
    ) 