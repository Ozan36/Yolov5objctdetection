import argparse
import cv2
import torch
from pathlib import Path
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_boxes, xyxy2xywh
from utils.plots import Annotator, colors

def detect_objects(
    model_path="yolov5s.pt",
    source="data/images/",
    img_size=608,
    conf_threshold=0.25,
    iou_threshold=0.45,
    device="cpu",
    view_result=False,
    save_result=True
):
    # Cihaz seçimi
    device = torch.device(device)
    
    # Modeli yükle
    model = DetectMultiBackend(model_path, device=device)
    model.eval()
    
    # Görüntü yükleyici
    dataset = LoadImages(source, img_size=img_size, auto=model.pt)
    
    # Her görüntü için işleme
    for path, img, img_original, _, _ in dataset:  # 5 değişkeni de alıyoruz
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
        for det in predictions:
            if len(det):
                # Kutuları orijinal görüntü boyutuna ölçekleme
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img_original.shape).round()
                
                # Görselleştirme
                annotator = Annotator(img_original, line_width=2)
                for *xyxy, conf, cls in det:
                    # Kutu ve etiket ekleme
                    label = f"{model.names[int(cls)]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(int(cls), True))
                
                # Sonucu gösterme veya kaydetme
                result = annotator.result()
                if view_result:
                    cv2.imshow("Detection", result)
                    cv2.waitKey(0)
                if save_result:
                    output_path = f"result_{Path(path).name}"
                    cv2.imwrite(output_path, result)
                    print(f"Sonuç kaydedildi: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov5s.pt", help="Model dosya yolu")
    parser.add_argument("--source", default="data/images", help="Giriş dosyası/dizini")
    parser.add_argument("--img-size", type=int, default=640, help="İşlem boyutu")
    parser.add_argument("--conf", type=float, default=0.25, help="Güven eşiği")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU eşiği")
    parser.add_argument("--device", default="cpu", help="cpu veya cuda")
    parser.add_argument("--view", action="store_true", help="Sonucu göster")
    parser.add_argument("--no-save", action="store_true", help="Sonucu kaydetme")
    args = parser.parse_args()
    
    detect_objects(
        model_path=args.model,
        source=args.source,
        img_size=args.img_size,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        view_result=args.view,
        save_result=not args.no_save
    )

if __name__ == "__main__":
    main()



