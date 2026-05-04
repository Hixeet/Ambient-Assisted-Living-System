import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from collections import deque
from datetime import datetime
import tensorflow as tf
import mediapipe as mp 
import os

# Konfigurasi Default
default_model_path = r"C:\Users\Lenovo\Documents\Skripsi\Percobaan Aplikasi implementasi - pose\augmentasi_pose_seq6_acc_98_.h5"
model = tf.keras.models.load_model(default_model_path, compile=False)
class_labels = ["Berdiri", "Berjalan", "Duduk", "Jatuh"]
IMG_SIZE = (128, 128)
IMAGE_HEIGHT, IMAGE_WIDTH = 128, 128
SEQUENCE_LENGTH = 6
frame_buffer = deque(maxlen=SEQUENCE_LENGTH)

# Pose Landmark Setup
mp_pose = mp.solutions.pose

# Landmark grouping definitions
HEAD_LANDMARKS       = list(range(0, 11))
BODY_LANDMARKS       = [11, 12]
LEFT_HAND_LANDMARKS  = [13, 15, 17, 19, 21]
RIGHT_HAND_LANDMARKS = [14, 16, 18, 20, 22]
LEFT_LEG_LANDMARKS   = [23, 25, 27, 29, 31]
RIGHT_LEG_LANDMARKS  = [24, 26, 28, 30, 32]

BLOCKED_CONNECTION_GROUPS = []

GROUP_MARKER = {
    'head': cv2.MARKER_TRIANGLE_UP, 'body': cv2.MARKER_DIAMOND,
    'left_hand': cv2.MARKER_TRIANGLE_DOWN, 'right_hand': cv2.MARKER_STAR,
    'left_leg': cv2.MARKER_SQUARE, 'right_leg': cv2.MARKER_TILTED_CROSS,
}
GROUP_COLOR = {
    'head': (0,255,255), 'body': (255,0,0),
    'left_hand': (0,255,0), 'right_hand': (0,200,0),
    'left_leg': (0,0,255), 'right_leg': (255,0,255),
}

def get_landmark_group(idx):
    if idx in HEAD_LANDMARKS:       return 'head'
    if idx in BODY_LANDMARKS:       return 'body'
    if idx in LEFT_HAND_LANDMARKS:  return 'left_hand'
    if idx in RIGHT_HAND_LANDMARKS: return 'right_hand'
    if idx in LEFT_LEG_LANDMARKS:   return 'left_leg'
    if idx in RIGHT_LEG_LANDMARKS:  return 'right_leg'
    return 'unknown'

def compute_head_centroid(landmarks, w, h):
    xs = [landmarks[i].x * w for i in HEAD_LANDMARKS if i < len(landmarks)]
    ys = [landmarks[i].y * h for i in HEAD_LANDMARKS if i < len(landmarks)]
    if not xs or not ys:
        return None
    return (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))

def draw_vertical_zigzag(canvas, amplitude=20, period=40, thickness=2):
    h, w = canvas.shape[:2]
    cx = w // 2
    pts, y, direction = [], 0, 1
    while y <= h:
        pts.append((cx + direction * amplitude, y))
        y += period
        direction *= -1
    for i in range(len(pts) - 1):
        cv2.line(canvas, pts[i], pts[i + 1], (0, 255, 255), thickness)

def draw_landmarks(canvas, landmarks, w, h):
    # khusus kepala → marker lebih kecil
    for i in HEAD_LANDMARKS:
        if i < len(landmarks):
            lm = landmarks[i]
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.drawMarker(canvas, (x, y), GROUP_COLOR['head'],
                           markerType=GROUP_MARKER['head'], markerSize=7, thickness=2)
    # sisanya
    for i, lm in enumerate(landmarks):
        if i in HEAD_LANDMARKS:
            continue
        x, y = int(lm.x * w), int(lm.y * h)
        g = get_landmark_group(i)
        cv2.drawMarker(canvas, (x, y), GROUP_COLOR.get(g, (128, 128, 128)),
                       markerType=GROUP_MARKER.get(g, cv2.MARKER_CROSS),
                       markerSize=15, thickness=2)

def draw_connections(canvas, landmarks, w, h):
    def interp(p1, p2, steps, marker, color):
        x1, y1 = int(p1.x * w), int(p1.y * h)
        x2, y2 = int(p2.x * w), int(p2.y * h)
        for t in range(1, steps):
            alpha = t / steps
            xi = int(x1 * (1 - alpha) + x2 * alpha)
            yi = int(y1 * (1 - alpha) + y2 * alpha)
            cv2.drawMarker(canvas, (xi, yi), color,
                           markerType=marker, markerSize=10, thickness=2)

    for a, b in mp_pose.POSE_CONNECTIONS:
        if a >= len(landmarks) or b >= len(landmarks):
            continue
        g1, g2 = get_landmark_group(a), get_landmark_group(b)
        if g1 == 'head' and g2 == 'head':
            continue
        if any(a in grp and b in grp for grp in BLOCKED_CONNECTION_GROUPS):
            continue
        p1, p2 = landmarks[a], landmarks[b]
        if g1 == g2 != 'unknown':
            interp(p1, p2, 5, GROUP_MARKER[g1], GROUP_COLOR[g1])
        elif (a, b) in [(11, 23), (12, 24)]:
            m = GROUP_MARKER.get(g1, cv2.MARKER_CROSS)
            c = GROUP_COLOR.get(g1, (128, 128, 128))
            interp(p1, p2, 9, m, c)

def crop_and_norm(frame, landmarks, w, h):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    if not xs or not ys:
        return None
    x1, y1 = int(min(xs) * w) - 20, int(min(ys) * h) - 20
    x2, y2 = int(max(xs) * w) + 20, int(max(ys) * h) + 20
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w), min(y2, h)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (IMAGE_HEIGHT, IMAGE_WIDTH)).astype(np.float32)/255.0

# Variabel Global 
previous_label = None
latest_activity = {'activity': 'Menunggu...', 'time': '-'}
activity_log = []
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                    enable_segmentation=False, min_detection_confidence=0.5)
current_sort_column = None
current_sort_reverse = False
APP_MODE = None
root = None
video_label = None
skeleton_label = None
label_aktivitas = None
label_waktu = None
tree = None
cap = None

# Variabel tambahan fitur baru
selected_camera_index = 0
selected_model_path = default_model_path

# ================== Fungsi Utama Pose ===================
def detect_pose_activity(frame):
    global previous_label, latest_activity, activity_log
    h, w = frame.shape[:2]
    now = datetime.now()
    str_now = now.strftime('%Y-%m-%d %H:%M:%S')
    skeleton = np.zeros_like(frame)

    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        draw_landmarks(skeleton, lm, w, h)
        draw_connections(skeleton, lm, w, h)

        norm = crop_and_norm(skeleton, lm, w, h)
        if norm is None:
            return skeleton

        frame_buffer.append(norm)
        if len(frame_buffer) == SEQUENCE_LENGTH:
            input_seq = np.expand_dims(np.array(frame_buffer), axis=0)
            preds = model.predict(input_seq, verbose=0)
            idx = np.argmax(preds)
            conf = np.max(preds) * 100
            label_only = class_labels[idx]
            activity_label = f"{label_only} ({conf:.2f}%)"

            if label_only != previous_label:
                if previous_label is not None and activity_log:
                    st = datetime.strptime(activity_log[-1]['start_time'], '%Y-%m-%d %H:%M:%S')
                    activity_log[-1]['duration'] = str(now - st).split('.')[0]

                activity_log.append({
                    'activity': activity_label,
                    'start_time': str_now,
                    'duration': '-'
                })
                previous_label = label_only

            latest_activity.update({
                'activity': activity_label,
                'time': str_now
            })

            try:
                label_aktivitas.config(text=f"Aktivitas: {activity_label}")
                label_waktu.config(text=f"Waktu: {str_now}")
                if label_only.lower() == "jatuh":
                    label_aktivitas.config(fg="#f44336")
                else:
                    label_aktivitas.config(fg="#4CAF50")
            except NameError:
                pass
    else:
        if previous_label != "no_pose":
            if previous_label and activity_log:
                st = datetime.strptime(activity_log[-1]['start_time'], '%Y-%m-%d %H:%M:%S')
                activity_log[-1]['duration'] = str(now - st).split('.')[0]

            activity_log.append({
                'activity': 'Tidak ada orang',
                'start_time': str_now,
                'duration': '-'
            })
            previous_label = "no_pose"

        latest_activity.update({
            'activity': 'Tidak ada orang',
            'time': str_now
        })

        try:
            label_aktivitas.config(text="Aktivitas: Tidak ada orang yang terdeteksi", fg="#888")
            label_waktu.config(text=f"Waktu: {str_now}")
        except NameError:
            pass

    return skeleton

# ================== Log & Sort ===================
def sort_log_by_column(col):
    global current_sort_column, current_sort_reverse
    if current_sort_column == col:
        current_sort_reverse = not current_sort_reverse
    else:
        current_sort_column = col
        current_sort_reverse = False
    refresh_log_view(force=True)

def refresh_log_view(force=False):
    sorted_entries = activity_log[:]
    if current_sort_column == 'Aktivitas':
        def sort_key(entry):
            label = entry['activity'].split('(')[0].strip().lower()
            is_jatuh = 'jatuh' in label
            return (not is_jatuh, label)
        sorted_entries = sorted(sorted_entries, key=sort_key, reverse=current_sort_reverse)
    elif current_sort_column == 'Waktu Mulai':
        def sort_key(entry):
            return datetime.strptime(entry['start_time'], '%Y-%m-%d %H:%M:%S')
        sorted_entries = sorted(sorted_entries, key=sort_key, reverse=current_sort_reverse)
    else:
        sorted_entries = list(reversed(activity_log))

    if tree is None:
        return

    try:
        y = tree.yview()
    except Exception:
        y = None

    for i in tree.get_children():
        tree.delete(i)
    for entry in sorted_entries:
        tag = 'abnormal' if entry['activity'].lower().startswith('jatuh') else ''
        tree.insert('', 'end',
                    values=(entry['activity'], entry['start_time'], entry['duration']),
                    tags=(tag,))
    if y:
        try:
            tree.yview_moveto(y[0])
        except Exception:
            pass

# Update Video 
def update_video():
    global cap
    if cap is None:
        return
    ret, frame = cap.read()
    if ret:
        skeleton = detect_pose_activity(frame)
        disp = frame.copy()
        img1 = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)))
        try:
            video_label.imgtk = img1
            video_label.config(image=img1)
        except Exception:
            pass
        if APP_MODE == 'developer' and skeleton_label is not None:
            try:
                img2 = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(skeleton, cv2.COLOR_BGR2RGB)))
                skeleton_label.imgtk = img2
                skeleton_label.config(image=img2)
            except Exception:
                pass
    refresh_log_view()
    root.after(30, update_video)

# Frame Main 
def show_main_frame(selected_mode):
    global APP_MODE, cap, model, selected_camera_index, selected_model_path
    APP_MODE = selected_mode
    frame_menu.pack_forget()
    build_main_ui()
    frame_main.pack(fill="both", expand=True)
    # load model
    try:
        model = tf.keras.models.load_model(selected_model_path, compile=False)
    except Exception as e:
        messagebox.showerror("Model", f"Gagal memuat model: {e}")
        back_to_menu()
        return
    cap = cv2.VideoCapture(selected_camera_index)
    if not cap.isOpened():
        messagebox.showerror("Kamera", f"Tidak dapat membuka kamera {selected_camera_index}.")
        back_to_menu()
        return
    root.after(0, update_video)

def back_to_menu():
    global cap, previous_label, frame_main
    if cap is not None:
        try: cap.release()
        except Exception: pass
        cap = None
    previous_label = None
    frame_main.pack_forget()
    frame_main.destroy()  # hapus frame main lama
    frame_main = tk.Frame(root, bg="#f2f2f2")  # buat ulang
    frame_menu.pack(fill="both", expand=True)

def save_log_to_file():
    if not activity_log:
        messagebox.showinfo("Simpan Log", "Belum ada log aktivitas untuk disimpan.")
        return
    filename = f"activity_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        for entry in activity_log:
            f.write(f"{entry['activity']} | {entry['start_time']} | {entry['duration']}\n")
    messagebox.showinfo("Simpan Log", f"Log aktivitas disimpan ke {filename}")

def build_main_ui():
    global video_label, skeleton_label, label_aktivitas, label_waktu, tree
    if getattr(frame_main, "_built", False): 
        return
    frame_main._built = True

    # Judul Aplikasi 
    root.title("Deteksi Aktivitas Pose")
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass

    # Header profesional
    header = tk.Frame(frame_main, bg="#1f2a3c", height=60)
    header.pack(fill="x")
    tk.Label(
        header, text="Deteksi Aktivitas Pose Real-time",
        font=("Segoe UI Semibold", 20), bg="#1f2a3c", fg="#ffffff"
    ).pack(side="left", padx=20)

    mode_label = tk.Label(header, text=f"Mode: {APP_MODE.capitalize() if APP_MODE else '-'}",
                          font=("Segoe UI", 12), bg="#1f2a3c", fg="#e0e0e0")
    mode_label.pack(side="right", padx=20)

    # Container utama
    container = tk.Frame(frame_main, bg="#f0f3f7")
    container.pack(fill="both", expand=True, padx=10, pady=10)

    # USER MODE 
    if APP_MODE == 'user':
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=3)
        container.grid_columnconfigure(1, weight=2)

        # Panel kiri: Video + Info
        left_panel = tk.Frame(container, bg="#ffffff", bd=0, highlightthickness=2,
                              highlightbackground="#d0d0d0", relief="flat")
        left_panel.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Shadow effect
        shadow = tk.Frame(container, bg="#d0d0d0")
        shadow.place(in_=left_panel, x=3, y=3, relwidth=1, relheight=1)
        left_panel.lift()

        video_label = tk.Label(left_panel, bg="#000000")
        video_label.pack(fill="both", expand=True, padx=5, pady=5)

        info_box = tk.Frame(left_panel, bg="#f7f8fa", padx=12, pady=8)
        info_box.pack(pady=10, fill="x", padx=5)

        tk.Label(
            info_box, text="Aktivitas Terdeteksi:",
            font=("Segoe UI Semibold", 12), bg="#f7f8fa", fg="#333"
        ).pack(anchor='w')

        label_aktivitas = tk.Label(
            info_box, text="Aktivitas: Menunggu...",
            font=("Segoe UI", 14), fg="#4CAF50", bg="#f7f8fa"
        )
        label_aktivitas.pack(anchor='w', pady=2)

        label_waktu = tk.Label(
            info_box, text="Waktu: -",
            font=("Segoe UI", 10), fg="#555", bg="#f7f8fa"
        )
        label_waktu.pack(anchor='w', pady=2)

        skeleton_label = None

        build_log_panel(container, 1)

    # DEVELOPER MODE
    elif APP_MODE == 'developer':
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=3)
        container.grid_columnconfigure(1, weight=2)
        container.grid_columnconfigure(2, weight=2)

        # Panel kiri: Video + Info
        left_panel = tk.Frame(container, bg="#ffffff", bd=0, highlightthickness=2,
                              highlightbackground="#d0d0d0", relief="flat")
        left_panel.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Shadow effect
        shadow = tk.Frame(container, bg="#d0d0d0")
        shadow.place(in_=left_panel, x=3, y=3, relwidth=1, relheight=1)
        left_panel.lift()

        video_label = tk.Label(left_panel, bg="#000000")
        video_label.pack(fill="both", expand=True, padx=5, pady=5)

        info_box = tk.Frame(left_panel, bg="#f7f8fa", padx=12, pady=8)
        info_box.pack(pady=10, fill="x", padx=5)

        tk.Label(
            info_box, text="Aktivitas Terdeteksi:",
            font=("Segoe UI Semibold", 12), bg="#f7f8fa", fg="#333"
        ).pack(anchor='w')

        label_aktivitas = tk.Label(
            info_box, text="Aktivitas: Menunggu...",
            font=("Segoe UI", 14), fg="#4CAF50", bg="#f7f8fa"
        )
        label_aktivitas.pack(anchor='w', pady=2)

        label_waktu = tk.Label(
            info_box, text="Waktu: -",
            font=("Segoe UI", 10), fg="#555", bg="#f7f8fa"
        )
        label_waktu.pack(anchor='w', pady=2)

        # Panel tengah: Skeleton
        skeleton_panel = tk.Frame(container, bg="#ffffff", bd=0, highlightthickness=2,
                                  highlightbackground="#d0d0d0", relief="flat")
        skeleton_panel.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        tk.Label(
            skeleton_panel, text="Skeleton Mediapipe",
            font=("Segoe UI Semibold", 14), bg="#ffffff", fg="#1f2a3c"
        ).pack(pady=5)

        skeleton_label = tk.Label(skeleton_panel, bg="#000000")
        skeleton_label.pack(fill="both", expand=True, padx=5, pady=5)

        build_log_panel(container, 2)


def build_log_panel(parent, col_index):
    global tree
    right_panel = tk.Frame(parent, bg="#ffffff", bd=0, highlightthickness=2,
                           highlightbackground="#d0d0d0", relief="flat")
    right_panel.grid(row=0, column=col_index, padx=10, pady=10, sticky="nsew")

    tk.Label(
        right_panel, text="📜 Riwayat Aktivitas",
        font=("Segoe UI Semibold", 14), bg="#ffffff", fg="#1f2a3c"
    ).pack(padx=8, pady=8, anchor='w')

    tree_frame = tk.Frame(right_panel, bg="#ffffff")
    tree_frame.pack(fill="both", expand=True, padx=5, pady=5)

    columns = ('Aktivitas', 'Waktu Mulai', 'Durasi')
    tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=18)
    for col in columns:
        tree.heading(col, text=col, command=lambda c=col: sort_log_by_column(c))
        tree.column(col, width=140, anchor='center')
    tree.tag_configure('abnormal', background='#ffe0e0', foreground='#b20000')
    tree.pack(side='left', fill='both', expand=True)

    scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
    scrollbar.pack(side='right', fill='y')
    tree.configure(yscrollcommand=scrollbar.set)

    # Tombol modern flat dengan hover effect
    btn_frame = tk.Frame(right_panel, bg="#ffffff")
    btn_frame.pack(fill="x", padx=5, pady=(8,5))

    def on_enter(e):
        e.widget.config(bg="#e0e0e0")
    def on_leave(e):
        e.widget.config(bg="#f0f0f0")

    save_btn = tk.Button(btn_frame, text="💾 Simpan Log", command=save_log_to_file,
                         bg="#f0f0f0", relief="flat")
    save_btn.pack(side="right", padx=5)
    save_btn.bind("<Enter>", on_enter)
    save_btn.bind("<Leave>", on_leave)

    back_btn = tk.Button(btn_frame, text="⬅ Kembali ke Menu", command=back_to_menu,
                         bg="#f0f0f0", relief="flat")
    back_btn.pack(side="right", padx=5)
    back_btn.bind("<Enter>", on_enter)
    back_btn.bind("<Leave>", on_leave)

# Frame Menu
def shade_color(hex_color, percent):
    """ Menggelapkan atau mencerahkan warna untuk hover efek """
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    r = max(0, min(255, r + int(r * percent / 100)))
    g = max(0, min(255, g + int(g * percent / 100)))
    b = max(0, min(255, b + int(b * percent / 100)))

    return f"#{r:02x}{g:02x}{b:02x}"

def fade_in(widget, delay=20, step=0.05):
    """ Animasi fade in untuk widget """
    alpha = 0
    def _fade():
        nonlocal alpha
        alpha += step
        if alpha > 1:
            alpha = 1
        try:
            widget.tk.call(widget._w, 'attributes', '-alpha', alpha)
        except:
            pass
        if alpha < 1:
            widget.after(delay, _fade)
    _fade()

def create_shadowed_button(parent, text, command, bg_color, fg_color):
    """ Tombol modern dengan shadow dan hover smooth """
    # Shadow: cukup lebar agar tombol tidak terlalu sempit
    shadow_width = 275  # lebar shadow yang cukup untuk teks
    shadow_height = 20  # tinggi shadow
    shadow = tk.Label(parent, bg="#a0a0a0", width=shadow_width//10, height=shadow_height//10)
    shadow.pack_propagate(0)
    shadow.pack(side="left", padx=10, pady=10)

    # Button
    btn = tk.Button(
        parent, text=text, command=command,
        bg=bg_color, fg=fg_color,
        font=("Segoe UI Semibold", 12),
        bd=0, relief="flat",
        activebackground=shade_color(bg_color, -10),
        padx=20, pady=10  # padding internal agar teks cukup ruang
    )
    btn.place(in_=shadow, x=0, y=0, relwidth=1, relheight=1)

    # Hover effect
    def on_enter(e):
        btn.configure(bg=shade_color(bg_color, -15))
    def on_leave(e):
        btn.configure(bg=bg_color)
    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)

    return btn

# Build Menu
def build_menu_ui():
    global frame_menu, frame_main, camera_combo, model_combo
    frame_menu = tk.Frame(root, bg="#f0f1f5")
    frame_menu.pack(fill="both", expand=True)
    frame_main = tk.Frame(root, bg="#f2f2f2")

    # Header
    title_label = tk.Label(
        frame_menu, text="Menu Utama",
        font=("Segoe UI", 28, "bold"),
        bg="#f0f1f5", fg="#1f1f2e"
    )
    title_label.pack(pady=(30,20))

    # Semi-transparent panel untuk pilihan
    select_panel = tk.Frame(frame_menu, bg="#ffffff")
    select_panel.pack(pady=10, padx=50)
    select_panel.configure(bg="#ffffff")
    select_panel.attributes = {}  # dummy, untuk konsistensi future alpha

    select_frame = tk.Frame(select_panel, bg="#ffffff")
    select_frame.pack(pady=10, padx=10)

    # Pilih kamera
    tk.Label(select_frame, text="Pilih Kamera:", font=("Segoe UI", 12),
             bg="#ffffff", fg="#333333").grid(row=0, column=0, sticky="w", padx=5, pady=5)
    camera_combo = ttk.Combobox(select_frame, state="readonly", width=30)
    camera_combo.grid(row=0, column=1, padx=5, pady=5)
    refresh_camera_list()

    # Pilih model
    tk.Label(select_frame, text="Pilih Model:", font=("Segoe UI", 12),
             bg="#ffffff", fg="#333333").grid(row=1, column=0, sticky="w", padx=5, pady=5)
    model_combo = ttk.Combobox(select_frame, state="readonly", width=30)
    model_combo.grid(row=1, column=1, padx=5, pady=5)
    refresh_model_list()

    # Tombol
    btn_frame = tk.Frame(frame_menu, bg="#f0f1f5")
    btn_frame.pack(pady=30)

    user_btn = create_shadowed_button(
        btn_frame, "Mulai (User Mode)", 
        lambda: start_with_selection("user"),
        bg_color="#4CAF50", fg_color="white"
    )

    dev_btn = create_shadowed_button(
        btn_frame, "Mulai (Developer Mode)", 
        lambda: start_with_selection("developer"),
        bg_color="#2196F3", fg_color="white"
    )

    # Fade-in tombol
    fade_in(user_btn)
    fade_in(dev_btn)

    # Rounded combobox
    for combo in [camera_combo, model_combo]:
        combo.configure(font=("Segoe UI", 11))
        combo.configure(background="#ffffff")
 
def refresh_camera_list():
    available = []
    for i in range(4):
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            available.append(f"Kamera {i}")
        cap.release()
    if not available:
        available = ["Tidak ada kamera terdeteksi"]
    camera_combo['values'] = available
    camera_combo.current(0)

def refresh_model_list():
    models = []
    default_dir = os.path.dirname(default_model_path)
    for file in os.listdir(default_dir):
        if file.endswith(".h5"):
            models.append(file)
    if not models:
        models = [os.path.basename(default_model_path)]
    model_combo['values'] = models
    model_combo.current(0)

def start_with_selection(mode):
    global selected_camera_index, selected_model_path
    if camera_combo.get().startswith("Kamera"):
        selected_camera_index = int(camera_combo.get().split()[-1])
    else:
        selected_camera_index = 0
    model_file = model_combo.get()
    selected_model_path = os.path.join(os.path.dirname(default_model_path), model_file)
    show_main_frame(mode)

# Main
if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1300x720")
    root.configure(bg="#e6e9f0")
    build_menu_ui()
    root.mainloop()
