import cv2, os, subprocess, numpy as np, time, threading, pickle, tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from insightface.app import FaceAnalysis
from datetime import datetime
from scipy.spatial import distance as dist
from collections import deque

class HyperFluxAI:
    def __init__(self, root):
        self.root = root
        self.root.title("V28 HYPER-FLUX | FULL INTELLIGENCE")
        self.root.attributes('-fullscreen', True)
        self.root.configure(bg="#000")

        # 1. SİSTEM TEMİZLİĞİ
        os.system("sudo pkill -9 rpicam && sudo pkill -9 libcamera")
        
        # 2. DEĞİŞKENLER VE AYARLAR
        self.running = True
        self.frames = {0: None, 1: None}
        self.display_frame = None
        self.lock = threading.Lock()
        
        # Bildirim Sistemi
        self.active_notifications = []
        self.max_notifications = 4

        # Yoklama Sistemi (şu an içeride kabul edilen öğrenciler)
        self.attended_students = set()
        
        # Takip ve Mantıksal Ayarlar
        self.next_object_id = 0
        self.tracked_persons = {} 
        self.max_disappeared = 20   # Artırıldı - daha stabil takip için
        self.line_x = 640           # 1280px genişliğin ortası (birleştirilmiş görüntü)
        self.buffer_zone = 50       # Buffer zone genişletildi
        
        # Yüz tanıma ayarları
        self.recognition_threshold = 0.50  # Threshold artırıldı - daha güvenilir tanıma
        self.min_recognition_confidence = 0.40  # Minimum güven eşiği
        
        # Her kişi için tanıma geçmişi (smoothing için)
        self.recognition_history = {}  # {oid: deque([name1, name2, ...])}
        self.history_size = 5  # Son 5 frame'deki tanıma sonuçlarını sakla
        
        # 3. AI MODELİNİ BAŞLAT
        self.face_app = None
        self.db = []
        threading.Thread(target=self.init_ai, daemon=True).start()

        # 4. ARAYÜZÜ OLUŞTUR
        self.build_gui()
        
        # Bildirim Konteynırı (Sağ Alt Köşe)
        self.notif_frame = tk.Frame(self.root, bg="#000", bd=0)
        self.notif_frame.place(relx=0.99, rely=0.99, anchor="se")

        # 5. MOTORLARI ÇALIŞTIR
        threading.Thread(target=self.camera_engine, args=(0,), daemon=True).start()
        threading.Thread(target=self.camera_engine, args=(1,), daemon=True).start()
        threading.Thread(target=self.core_processor, daemon=True).start()
        
        self.ui_update_loop()

    def init_ai(self):
        try:
            model = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            model.prepare(ctx_id=0, det_size=(320, 320), det_thresh=0.45)
            self.face_app = model
            if os.path.exists("v7_pro_plus.pickle"):
                with open("v7_pro_plus.pickle", "rb") as f:
                    self.db = pickle.load(f)
                    # Veritabanındaki embeddingleri normalize et
                    for p in self.db:
                        if "embedding" in p:
                            p["embedding"] = p["embedding"] / np.linalg.norm(p["embedding"])
            print("AI ve Veritabanı Entegre Edildi.")
        except Exception as e:
            print(f"AI Hatası: {e}")

    def build_gui(self):
        # Sol Panel: Video
        self.v_label = tk.Label(self.root, bg="black")
        self.v_label.pack(side="left", fill="both", expand=True, padx=10)

        # Sağ Panel: Tablo ve Saat
        self.side_panel = tk.Frame(self.root, bg="#020617", width=420)
        self.side_panel.pack(side="right", fill="y")

        self.clock_label = tk.Label(self.side_panel, text="--:--:--", fg="#38bdf8", bg="#020617", font=("Arial", 28, "bold"))
        self.clock_label.pack(pady=25)

        tk.Label(self.side_panel, text="TAKİP VE GEÇİŞ LOGLARI", fg="#22c55e", bg="#020617", font=("Arial", 10, "bold")).pack()
        
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background="#0f172a", foreground="white", fieldbackground="#0f172a", rowheight=35)
        
        self.tree = ttk.Treeview(self.side_panel, columns=("N", "M", "T"), show="headings")
        self.tree.heading("N", text="HEDEF"); self.tree.heading("M", text="EYLEM"); self.tree.heading("T", text="SAAT")
        self.tree.column("N", width=100); self.tree.column("M", width=180); self.tree.column("T", width=80)
        self.tree.pack(padx=10, pady=10, fill="both", expand=True)

    def show_toast(self, name, action=""):
        """Sağ alt köşede modern bir bildirim kartı oluşturur."""
        # Bildirim sayısını sınırla
        if len(self.active_notifications) >= self.max_notifications:
            try:
                oldest = self.active_notifications.pop(0)
                oldest.destroy()
            except: pass

        is_known = (name != "BELIRSIZ")
        bg_color = "#0f172a"  # Koyu lacivert/siyah
        accent_color = "#22c55e" if is_known else "#ef4444" # Yeşil veya Kırmızı
        icon = "👤" if is_known else "⚠️"
        
        # Ana kart çerçevesi
        toast = tk.Frame(self.notif_frame, bg=bg_color, pady=8, padx=12, highlightthickness=1, highlightbackground=accent_color)
        toast.pack(pady=4, fill="x", anchor="e")
        
        # İçerik düzeni
        top_frame = tk.Frame(toast, bg=bg_color)
        top_frame.pack(fill="x")
        
        tk.Label(top_frame, text=icon, fg=accent_color, bg=bg_color, font=("Arial", 14)).pack(side="left", padx=(0, 8))
        
        text_frame = tk.Frame(top_frame, bg=bg_color)
        text_frame.pack(side="left", fill="both")
        
        # İsim ve Durum
        display_name = "BİLİNMEYEN" if name == "BELIRSIZ" else name
        tk.Label(text_frame, text=display_name, fg="white", bg=bg_color, font=("Arial", 10, "bold")).pack(anchor="w")
        
        status_text = action if action else ("SİSTEME GİRİŞ" if is_known else "YABANCI TESPİTİ")
        tk.Label(text_frame, text=status_text, fg="#94a3b8", bg=bg_color, font=("Arial", 8)).pack(anchor="w")
        
        # Zaman Damgası
        tk.Label(toast, text=datetime.now().strftime("%H:%M:%S"), fg="#475569", bg=bg_color, font=("Arial", 7)).pack(anchor="e")

        self.active_notifications.append(toast)

        # 6 saniye sonra otomatik sil
        def remove():
            if toast.winfo_exists():
                if toast in self.active_notifications:
                    self.active_notifications.remove(toast)
                toast.destroy()
        
        self.root.after(6000, remove)

    def log_event(self, name, action):
        now = datetime.now().strftime("%H:%M:%S")
        self.root.after(0, lambda: self.tree.insert("", 0, values=(name, action, now)))

    def camera_engine(self, cam_id):
        cmd = ['rpicam-vid', '-t', '0', '--camera', str(cam_id), '--width', '640', '--height', '480', '--framerate', '25', '--codec', 'mjpeg', '-n', '-o', '-']
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**6)
            raw = b''
            while self.running:
                chunk = p.stdout.read(4096)
                if not chunk: break
                raw += chunk
                a, b = raw.find(b'\xff\xd8'), raw.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    img = cv2.imdecode(np.frombuffer(raw[a:b+2], dtype=np.uint8), cv2.IMREAD_COLOR)
                    raw = raw[b+2:]
                    if img is not None:
                        with self.lock: self.frames[cam_id] = img
        except: pass

    def get_identity(self, embedding):
        """Geliştirilmiş yüz tanıma - daha güvenilir sonuçlar için"""
        if not self.db or self.face_app is None: 
            return "BELIRSIZ", 0.0
        
        # Embedding'i normalize et
        norm_emb = embedding / np.linalg.norm(embedding)
        
        max_score = -1
        identity = "BELIRSIZ"
        
        for p in self.db:
            if "embedding" not in p:
                continue
            # Cosine similarity
            score = np.dot(norm_emb, p["embedding"])
            
            if score > max_score:
                max_score = score
                identity = p["name"]
        
        # Threshold kontrolü
        if max_score >= self.recognition_threshold:
            return identity, max_score
        elif max_score >= self.min_recognition_confidence:
            # Düşük güven - ama yine de döndür (smoothing ile iyileştirilebilir)
            return identity, max_score
        else:
            return "BELIRSIZ", max_score

    def update_recognition_history(self, oid, name):
        """Tanıma geçmişini güncelle - smoothing için"""
        if oid not in self.recognition_history:
            self.recognition_history[oid] = deque(maxlen=self.history_size)
        self.recognition_history[oid].append(name)
        
        # En sık görülen ismi döndür
        if len(self.recognition_history[oid]) > 0:
            names = list(self.recognition_history[oid])
            # En sık görülen ismi bul
            most_common = max(set(names), key=names.count)
            return most_common
        return name

    def get_smoothed_name(self, oid):
        """Geçmişe dayalı yumuşatılmış isim"""
        if oid in self.recognition_history and len(self.recognition_history[oid]) > 0:
            names = list(self.recognition_history[oid])
            # BELIRSIZ'leri filtrele ve en sık görülen ismi bul
            known_names = [n for n in names if n != "BELIRSIZ"]
            if len(known_names) > 0:
                most_common = max(set(known_names), key=known_names.count)
                # Eğer en az 3 kez görüldüyse güvenilir kabul et
                if known_names.count(most_common) >= 3:
                    return most_common
        return "BELIRSIZ"

    def core_processor(self):
        while self.running:
            if self.face_app is None:
                time.sleep(1); continue

            with self.lock:
                f0, f1 = self.frames[0], self.frames[1]
                if f0 is not None and f1 is not None:
                    combined = np.hstack((f0, f1))
                elif f0 is not None:
                    combined = np.hstack((f0, np.zeros_like(f0)))
                else:
                    time.sleep(0.1); continue

            rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            faces = self.face_app.get(rgb)
            
            input_centroids = []
            face_meta = {}

            for f in faces:
                bbox = f.bbox.astype(int)
                cX, cY = int((bbox[0] + bbox[2]) / 2.0), int((bbox[1] + bbox[3]) / 2.0)
                input_centroids.append((cX, cY))
                name, confidence = self.get_identity(f.embedding)
                face_meta[(cX, cY)] = {
                    "bbox": bbox, 
                    "name": name, 
                    "confidence": confidence,
                    "embedding": f.embedding
                }

            # Takip güncellemesi
            if len(input_centroids) == 0:
                # Hiç yüz yok - tüm tracked kişiler için disappeared artır
                for oid in list(self.tracked_persons.keys()):
                    self.tracked_persons[oid]["disappeared"] += 1
                    if self.tracked_persons[oid]["disappeared"] > self.max_disappeared:
                        del self.tracked_persons[oid]
                        if oid in self.recognition_history:
                            del self.recognition_history[oid]
            else:
                if not self.tracked_persons:
                    # İlk kişiler - direkt kaydet
                    for c in input_centroids:
                        self.register_person(c, face_meta[c])
                else:
                    self.update_tracking_logic(input_centroids, face_meta)

            with self.lock:
                self.display_frame = combined.copy()
            time.sleep(0.02)

    def register_person(self, centroid, meta):
        """Yeni kişi kaydet"""
        side = self.determine_side(centroid[0])
        oid = self.next_object_id
        self.tracked_persons[oid] = {
            "centroid": centroid, 
            "bbox": meta["bbox"], 
            "name": meta["name"],
            "disappeared": 0, 
            "side": side,
            "last_x": centroid[0],  # Geçiş tespiti için son X pozisyonu
            "crossed_line": False,  # Çizgiyi geçti mi?
            "confidence": meta.get("confidence", 0.0)
        }
        # Tanıma geçmişini başlat
        self.update_recognition_history(oid, meta["name"])
        self.next_object_id += 1

    def determine_side(self, x):
        """X koordinatına göre bölge belirle"""
        if x < (self.line_x - self.buffer_zone):
            return "A"
        elif x > (self.line_x + self.buffer_zone):
            return "B"
        else:
            # Buffer zone içinde - önceki bölgeyi koru veya varsayılan olarak A
            return "A"  # Varsayılan

    def update_tracking_logic(self, input_centroids, face_meta):
        """Geliştirilmiş takip mantığı - daha stabil ID takibi"""
        object_ids = list(self.tracked_persons.keys())
        object_centroids = [self.tracked_persons[oid]["centroid"] for oid in object_ids]
        
        if len(object_centroids) == 0 or len(input_centroids) == 0:
            return
        
        # Mesafe matrisi
        D = dist.cdist(np.array(object_centroids), np.array(input_centroids))
        
        # Minimum mesafe eşiği (piksel cinsinden)
        max_distance = 150  # Maksimum mesafe eşiği
        
        # Hungarian algorithm benzeri eşleştirme
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols: 
                continue
            
            distance = D[row, col]
            if distance > max_distance:
                continue  # Çok uzak - eşleştirme
            
            oid = object_ids[row]
            new_c = input_centroids[col]
            meta = face_meta[new_c]
            
            # Eski bilgileri al
            old_x = self.tracked_persons[oid]["last_x"]
            old_side = self.tracked_persons[oid]["side"]
            old_name = self.tracked_persons[oid]["name"]
            
            # Yeni bilgileri güncelle
            self.tracked_persons[oid]["centroid"] = new_c
            self.tracked_persons[oid]["bbox"] = meta["bbox"]
            self.tracked_persons[oid]["disappeared"] = 0
            self.tracked_persons[oid]["last_x"] = new_c[0]
            
            # Yüz tanıma güncellemesi - her zaman güncelle (smoothing ile)
            new_name, confidence = self.get_identity(meta["embedding"])
            self.tracked_persons[oid]["confidence"] = confidence
            
            # Tanıma geçmişini güncelle
            smoothed_name = self.update_recognition_history(oid, new_name)
            
            # Eğer smoothed name güvenilirse kullan
            if smoothed_name != "BELIRSIZ":
                self.tracked_persons[oid]["name"] = smoothed_name
            elif confidence >= self.min_recognition_confidence:
                self.tracked_persons[oid]["name"] = new_name
            # Eğer eski isim BELIRSIZ değilse ve yeni tanıma başarısızsa, eski ismi koru
            elif old_name != "BELIRSIZ":
                self.tracked_persons[oid]["name"] = old_name

            # Bölge ve geçiş kontrolü
            new_side = self.determine_side(new_c[0])
            
            # Geçiş tespiti - çizgiyi geçti mi?
            crossed = False
            direction = None
            
            # Eğer buffer zone dışındaysa ve çizgiyi geçtiyse
            if abs(new_c[0] - self.line_x) > self.buffer_zone:
                # A'dan B'ye geçiş
                if old_x < self.line_x and new_c[0] > self.line_x:
                    crossed = True
                    direction = "A->B"
                    new_side = "B"
                # B'den A'ya geçiş
                elif old_x > self.line_x and new_c[0] < self.line_x:
                    crossed = True
                    direction = "B->A"
                    new_side = "A"
            
            # Eğer geçiş tespit edildiyse
            if crossed and not self.tracked_persons[oid]["crossed_line"]:
                self.tracked_persons[oid]["crossed_line"] = True
                self.tracked_persons[oid]["side"] = new_side
                
                p_name = self.tracked_persons[oid]["name"]
                if direction == "A->B":
                    label = "GİRİŞ (A->B)"
                    action_text = "odaya girdi"
                else:
                    label = "ÇIKIŞ (B->A)"
                    action_text = "odadan çıktı"

                # Geçiş olayı için log ve bildirim
                self.log_event(p_name, label)
                self.root.after(0, lambda n=p_name, a=action_text: self.show_toast(n, a))
            else:
                # Geçiş yoksa sadece bölgeyi güncelle
                self.tracked_persons[oid]["side"] = new_side
                # Eğer buffer zone dışına çıktıysa crossed_line'ı sıfırla (tekrar geçiş yapabilir)
                if abs(new_c[0] - self.line_x) > self.buffer_zone * 1.5:
                    self.tracked_persons[oid]["crossed_line"] = False

            # --- YOKLAMA MANTIĞI (bölge tabanlı) ---
            # İç bölgede görünenlerin yoklamasını al, dış bölgede görünen ve yoklaması olanları "sınıftan çıktı" yap.
            final_name = self.tracked_persons[oid]["name"]
            current_side = self.tracked_persons[oid]["side"]

            if final_name != "BELIRSIZ":
                # İç bölge (B): yoklama al (sadece ilk kez)
                if current_side == "B" and final_name not in self.attended_students:
                    self.attended_students.add(final_name)
                    self.log_event(final_name, "YOKLAMA ALINDI")
                    self.root.after(0, lambda n=final_name: self.show_toast(n, "yoklaması alındı"))

                # Dış bölge (A): daha önce yoklaması alınmışsa "sınıftan çıktı" olarak işaretle (sadece ilk çıkışta)
                elif current_side == "A" and final_name in self.attended_students:
                    self.attended_students.remove(final_name)
                    self.log_event(final_name, "SINIFTAN ÇIKTI")
                    self.root.after(0, lambda n=final_name: self.show_toast(n, "sınıftan çıktı"))
            # --- YOKLAMA MANTIĞI SONU ---

            used_rows.add(row)
            used_cols.add(col)

        # Kullanılmayan tracked kişiler için disappeared artır
        unused_rows = set(range(len(object_ids))).difference(used_rows)
        for row in unused_rows:
            oid = object_ids[row]
            self.tracked_persons[oid]["disappeared"] += 1
            if self.tracked_persons[oid]["disappeared"] > self.max_disappeared:
                del self.tracked_persons[oid]
                if oid in self.recognition_history:
                    del self.recognition_history[oid]

        # Yeni kişileri kaydet
        unused_cols = set(range(len(input_centroids))).difference(used_cols)
        for col in unused_cols:
            self.register_person(input_centroids[col], face_meta[input_centroids[col]])

    def ui_update_loop(self):
        self.clock_label.config(text=datetime.now().strftime("%H:%M:%S"))
        if self.display_frame is not None:
            canvas = self.display_frame.copy()
            h, w = canvas.shape[:2]

            # Orta çizgiyi çiz
            cv2.line(canvas, (self.line_x, 0), (self.line_x, h), (0, 255, 255), 2)
            # Buffer zone çizgileri
            cv2.line(canvas, (self.line_x - self.buffer_zone, 0), (self.line_x - self.buffer_zone, h), (100, 100, 100), 1)
            cv2.line(canvas, (self.line_x + self.buffer_zone, 0), (self.line_x + self.buffer_zone, h), (100, 100, 100), 1)
            
            cv2.putText(canvas, "DIS BOLGE (A)", (20, 40), 0, 0.8, (255, 255, 255), 2)
            cv2.putText(canvas, "IC BOLGE (B)", (w-150, 40), 0, 0.8, (255, 255, 255), 2)

            for oid, data in self.tracked_persons.items():
                if data["disappeared"] > 0: 
                    continue
                x1, y1, x2, y2 = data["bbox"]
                color = (34, 197, 94) if data["name"] != "BELIRSIZ" else (59, 130, 246)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                
                # İsim ve bölge bilgisi
                name_text = f"{data['name']} [{data['side']}]"
                cv2.putText(canvas, name_text, (x1, y1-10), 0, 0.5, color, 2)
                
                # Centroid'i göster
                cX, cY = data["centroid"]
                cv2.circle(canvas, (cX, cY), 5, color, -1)

            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            img_tk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
            self.v_label.config(image=img_tk)
            self.v_label.image = img_tk
        
        self.root.after(20, self.ui_update_loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = HyperFluxAI(root)
    root.bind("<Escape>", lambda e: root.destroy())
    root.mainloop()
