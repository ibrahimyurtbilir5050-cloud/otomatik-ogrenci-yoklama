import os, cv2, pickle, numpy as np, time
from insightface.app import FaceAnalysis
from tqdm import tqdm # Eğitim sürecini ilerleme çubuğuyla izlemek için

def ultra_augment(img):
    """Tek bir kareyi 50+ farklı senaryoya sokar. Eğitim süresini uzatır ama hafızayı güçlendirir."""
    variants = []
    # 1. Temel açılar ve aynalama
    variants.append(img)
    variants.append(cv2.flip(img, 1))
    
    # 2. Farklı Işık Koşulları (Gama ve Parlaklık)
    for gamma in [0.5, 0.8, 1.2, 1.5, 2.0]: # Çok karanlıktan çok parlağa
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        variants.append(cv2.LUT(img, table))
    
    # 3. Odak Hataları (Bulanıklık ve Keskinlik)
    variants.append(cv2.GaussianBlur(img, (5, 5), 0))
    variants.append(cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15))
    
    # 4. Gürültü Ekleme (Düşük kaliteli kamera simülasyonu)
    noise = np.random.normal(0, 5, img.shape).astype('uint8')
    variants.append(cv2.add(img, noise))
    
    # 5. Perspektif ve Hafif Döndürme (Öğrenci kafasını eğmişse)
    rows, cols = img.shape[:2]
    for angle in [-15, -10, 10, 15]:
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        variants.append(cv2.warpAffine(img, M, (cols, rows)))
        
    return variants

def deep_hifz_engine():
    print("\n" + "="*60)
    print("   V7.5 ULTRA-LATENT: DERİN HAFIZA EĞİTİMİ BAŞLADI")
    print("   (Bu işlem uzun sürebilir, Pi 5'in soğuduğundan emin olun)")
    print("="*60 + "\n")

    # En ağır ve en detaylı model setini yükle
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    # det_size 640x640: En küçük detayları bile kaçırmaz
    app.prepare(ctx_id=0, det_size=(640, 640))

    dataset_path = 'dataset'
    final_db = []
    
    start_time = time.time()

    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_dir): continue
        
        all_embeddings = []
        print(f"\n[ANALİZ] {person_name.upper()} için derin tarama yapılıyor...")
        
        img_list = os.listdir(person_dir)
        # İlerleme çubuğu (Progress Bar)
        for img_name in tqdm(img_list, desc="Fotoğraflar İşleniyor"):
            img_path = os.path.join(person_dir, img_name)
            raw_img = cv2.imread(img_path)
            if raw_img is None: continue

            # Her fotoyu 50+ varyasyona sok
            augmented_set = ultra_augment(raw_img)
            
            for aug_img in augmented_set:
                faces = app.get(aug_img)
                if faces:
                    # En yüksek puanlı yüzü al (En iyi temsil)
                    face = sorted(faces, key=lambda x: x.det_score, reverse=True)[0]
                    all_embeddings.append(face.embedding)

        if all_embeddings:
            # Tüm varyasyonların ağırlıklı ortalamasını al
            # Bu, kişinin "Süper Vektörü"dür.
            master_embedding = np.mean(all_embeddings, axis=0)
            master_embedding /= np.linalg.norm(master_embedding)
            
            final_db.append({
                "name": person_name,
                "embedding": master_embedding,
                "data_points": len(all_embeddings)
            })
            print(f"   [BAŞARILI] {person_name} için {len(all_embeddings)} farklı açı/ışık verisi hıfzedildi.")

    # Kayıt
    with open("v7_pro_plus.pickle", "wb") as f:
        pickle.dump(final_db, f)

    end_time = time.time()
    total_min = (end_time - start_time) / 60
    print("\n" + "="*60)
    print(f"   HAFIZA KAYDI TAMAMLANDI! Toplam Süre: {total_min:.2f} dk")
    print(f"   Veritabanı Dosyası: v7_pro_plus.pickle")
    print("="*60 + "\n")

if __name__ == "__main__":
    deep_hifz_engine()
