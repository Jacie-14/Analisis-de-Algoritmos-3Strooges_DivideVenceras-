import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageFilter
import numpy as np

# ---------- util: histograma y alineacion previa ----------

def match_histogram(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Ajusta el histograma de 'src' para que coincida con el de 'ref'.
    Entradas y salidas en 0..255 (float32).
    """
    s = src.clip(0, 255).astype(np.uint8).ravel()
    r = ref.clip(0, 255).astype(np.uint8).ravel()

    s_hist = np.bincount(s, minlength=256).astype(np.float64)
    r_hist = np.bincount(r, minlength=256).astype(np.float64)
    s_cdf = np.cumsum(s_hist) / s_hist.sum()
    r_cdf = np.cumsum(r_hist) / r_hist.sum()

    inv = np.interp(s_cdf, r_cdf, np.arange(256))
    out = inv[s].reshape(src.shape)
    return out.astype(np.float32)

def coarse_align_by_shift(A: np.ndarray, B: np.ndarray, max_shift: int = 8, step: int = 1):
    """
    Busca desplazamiento (dx, dy) entero que minimiza el MSE entre A y B.
    Devuelve B_shifted, dx, dy.
    """
    H, W = A.shape

    def blur(im):
        # kernel [1,2,1]/4 en x e y
        t = np.pad(im, ((0,0),(1,1)), mode="edge")
        imx = (t[:, :-2] + 2*t[:, 1:-1] + t[:, 2:]) * 0.25
        t = np.pad(imx, ((1,1),(0,0)), mode="edge")
        imy = (t[:-2, :] + 2*t[1:-1, :] + t[2:, :]) * 0.25
        return imy

    Ab = blur(A); Bb = blur(B)

    best = (1e18, 0, 0)
    for dy in range(-max_shift, max_shift+1, step):
        y0 = max(0, dy); y1 = min(H, H+dy); Y0 = -dy if dy < 0 else 0
        for dx in range(-max_shift, max_shift+1, step):
            x0 = max(0, dx); x1 = min(W, W+dx); X0 = -dx if dx < 0 else 0
            if y1-y0 < H//2 or x1-x0 < W//2:
                continue
            a = Ab[y0:y1, x0:x1]
            b = Bb[Y0:Y0+(y1-y0), X0:X0+(x1-x0)]
            mse = float(((a - b) ** 2).mean())
            if mse < best[0]:
                best = (mse, dx, dy)

    _, dx, dy = best
    B_shift = np.zeros_like(B)
    y0 = max(0, dy); y1 = min(H, H+dy); Y0 = -dy if dy < 0 else 0
    x0 = max(0, dx); x1 = min(W, W+dx); X0 = -dx if dx < 0 else 0
    B_shift[y0:y1, x0:x1] = B[Y0:Y0+(y1-y0), X0:X0+(x1-x0)]
    return B_shift, dx, dy

# ---------- util numerico ----------

def integral_image(a: np.ndarray) -> np.ndarray:
    return np.pad(a, ((1,0),(1,0)), mode="constant").cumsum(0).cumsum(1)

def box_sum(ii: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
    return float(ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0])

# ---------- GUI ----------

class ComparadorGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Comparacion adaptativa por regiones (Quadtree + SSIM + percentiles)")
        self.geometry("1100x680")
        self.minsize(960, 560)

        self.img_a = None
        self.img_b = None
        self.tk_a = None
        self.tk_b = None
        self.disp_a = None
        self.disp_b = None

        self.regions = []

        self._build_menu()
        self._build_layout()

    def _build_menu(self):
        menubar = tk.Menu(self)
        archivo = tk.Menu(menubar, tearoff=0)
        archivo.add_command(label="Cargar Imagen A...", command=lambda: self.load_image('A'))
        archivo.add_command(label="Cargar Imagen B...", command=lambda: self.load_image('B'))
        archivo.add_separator()
        archivo.add_command(label="Salir", command=self.destroy)
        menubar.add_cascade(label="Archivo", menu=archivo)
        self.config(menu=menubar)

    def _build_layout(self):
        container = ttk.Frame(self, padding=10); container.pack(fill="both", expand=True)

        left = ttk.LabelFrame(container, text="Imagen A", padding=10)
        left.pack(side="left", fill="both", expand=True, padx=(0,5))
        self.canvas_a = tk.Canvas(left, bg="#1f1f1f", highlightthickness=0); self.canvas_a.pack(fill="both", expand=True)

        right = ttk.LabelFrame(container, text="Imagen B", padding=10)
        right.pack(side="left", fill="both", expand=True, padx=(5,0))
        self.canvas_b = tk.Canvas(right, bg="#1f1f1f", highlightthickness=0); self.canvas_b.pack(fill="both", expand=True)

        bottom = ttk.Frame(self, padding=10); bottom.pack(side="bottom", fill="x")

        params = ttk.Frame(bottom); params.pack(side="left", anchor="w")

        ttk.Label(params, text="ssim_min").grid(row=0, column=0, sticky="w")
        self.ent_ssim = ttk.Entry(params, width=6); self.ent_ssim.insert(0, "0.97")
        self.ent_ssim.grid(row=0, column=1, padx=(4,12))

        ttk.Label(params, text="min_bloque").grid(row=0, column=2, sticky="w")
        self.ent_min = ttk.Entry(params, width=6); self.ent_min.insert(0, "8")
        self.ent_min.grid(row=0, column=3, padx=(4,12))

        ttk.Label(params, text="hp_radius").grid(row=0, column=4, sticky="w")
        self.ent_hp = ttk.Entry(params, width=4); self.ent_hp.insert(0, "8")
        self.ent_hp.grid(row=0, column=5, padx=(4,12))

        ttk.Label(params, text="delta_pix").grid(row=0, column=6, sticky="w")
        self.ent_dp = ttk.Entry(params, width=6); self.ent_dp.insert(0, "18")
        self.ent_dp.grid(row=0, column=7, padx=(4,12))

        self.modo_robusto = tk.BooleanVar(value=True)
        ttk.Checkbutton(params, text="robusto luz", variable=self.modo_robusto).grid(row=0, column=8, padx=(4,12))

        self.btn_compare = ttk.Button(bottom, text="Comparar (quadtree+SSIM)",
                                      state="disabled", command=self.compare_quadtree_ssim)
        self.btn_compare.pack(side="right", padx=(6,0))

        self.btn_clear = ttk.Button(bottom, text="Limpiar", state="disabled", command=self.clear_regions)
        self.btn_clear.pack(side="right")

        self.bind("<Configure>", self._on_resize)

    # ---------- carga y display ----------

    def load_image(self, side='A'):
        path = filedialog.askopenfilename(
            title=f"Selecciona Imagen {side}",
            filetypes=[("Imagenes", ".png;.jpg;.jpeg;.bmp;.tif;.tiff")]
        )
        if not path: return
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir la imagen:\n{e}")
            return

        if side == 'A':
            self.img_a = img; self._display_on_canvas(self.canvas_a, img, 'A')
        else:
            self.img_b = img; self._display_on_canvas(self.canvas_b, img, 'B')

        if self.img_a is not None and self.img_b is not None:
            self.btn_compare.config(state="normal")
            self.btn_clear.config(state="normal")

    def _display_on_canvas(self, canvas, pil_img, which='A'):
        cw = canvas.winfo_width() if canvas.winfo_width() > 2 else canvas.winfo_reqwidth()
        ch = canvas.winfo_height() if canvas.winfo_height() > 2 else canvas.winfo_reqheight()
        if cw <= 1 or ch <= 1:
            self.update_idletasks(); cw, ch = canvas.winfo_width(), canvas.winfo_height()
        img = pil_img.copy(); img.thumbnail((cw, ch))
        tk_img = ImageTk.PhotoImage(img)
        canvas.delete("all"); cx, cy = cw//2, ch//2
        canvas.create_image(cx, cy, image=tk_img, anchor="center")
        disp_w, disp_h = img.size; ox, oy = (cw-disp_w)//2, (ch-disp_h)//2
        if which == 'A':
            self.tk_a = tk_img; self.disp_a = {'w':disp_w,'h':disp_h,'ox':ox,'oy':oy,'canvas':canvas}
        else:
            self.tk_b = tk_img; self.disp_b = {'w':disp_w,'h':disp_h,'ox':ox,'oy':oy,'canvas':canvas}

    def _on_resize(self, _):
        if self.img_a: self._display_on_canvas(self.canvas_a, self.img_a, 'A')
        if self.img_b: self._display_on_canvas(self.canvas_b, self.img_b, 'B')
        if self.regions: self.draw_regions(self.regions)

    # ---------- preprocesado ----------

    def _to_highpass(self, pil_gray, radius=8):
        base = np.asarray(pil_gray, dtype=np.float32)
        blur = np.asarray(pil_gray.filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.float32)
        hp = base - blur
        hp -= hp.min()
        m = hp.max()
        if m > 0: hp *= 255.0 / m
        return hp.astype(np.float32)

    def _prep_two(self):
        if self.img_a is None or self.img_b is None:
            raise ValueError("Carga dos imagenes antes de comparar.")

        a = self.img_a.convert("L")
        b = self.img_b.convert("L")
        w, h = min(a.width, b.width), min(a.height, b.height)
        a = a.resize((w, h))
        b = b.resize((w, h))

        A = np.asarray(a, dtype=np.float32)
        B = np.asarray(b, dtype=np.float32)

        # igualar histograma B->A
        B = match_histogram(B, A)
        # pre-alinear por traslacion entera
        B, _, _ = coarse_align_by_shift(A, B, max_shift=8, step=1)

        # high-pass opcional (despues de alinear)
        try: hp_r = int(self.ent_hp.get())
        except: hp_r = 8
        hp_r = max(4, hp_r)
        if self.modo_robusto.get():
            A = self._to_highpass(Image.fromarray(A.astype(np.uint8)), radius=hp_r)
            B = self._to_highpass(Image.fromarray(B.astype(np.uint8)), radius=hp_r)

        return A, B

    # ---------- integrales y SSIM ----------

    def _build_integrals(self, A: np.ndarray, B: np.ndarray):
        Ai  = integral_image(A);  Bi  = integral_image(B)
        A2i = integral_image(A*A); B2i = integral_image(B*B)
        ABi = integral_image(A*B)
        return Ai, Bi, A2i, B2i, ABi

    def _ssim_region(self, Ai, Bi, A2i, B2i, ABi, x, y, w, h):
        area = w*h
        if area <= 0: return 1.0
        sa = box_sum(Ai,  x, y, x+w, y+h); ma = sa / area
        sb = box_sum(Bi,  x, y, x+w, y+h); mb = sb / area
        sa2 = box_sum(A2i, x, y, x+w, y+h); va = max(sa2/area - ma*ma, 1e-6)
        sb2 = box_sum(B2i, x, y, x+w, y+h); vb = max(sb2/area - mb*mb, 1e-6)
        sab = box_sum(ABi, x, y, x+w, y+h); cov = sab/area - ma*mb

        L = 255.0; K1 = 0.01; K2 = 0.03
        C1 = (K1*L)**2; C2 = (K2*L)**2
        num = (2*ma*mb + C1) * (2*cov + C2)
        den = (ma*ma + mb*mb + C1) * (va + vb + C2)
        ssim = num / den
        return float(max(0.0, min(1.0, ssim)))

    # ---------- quadtree con SSIM + percentiles ----------

    def _quadtree_ssim(self, Ai, Bi, A2i, B2i, ABi, A, B,
                       x, y, w, h, ssim_min, min_block, delta_pix, out_rects):
        """
        divide y venceras:
          - subdividir si ssim < ssim_min y hay pico local (p95 alto)
          - marcar hoja si ssim < ssim_min y p99 >= delta_pix (y algo de textura)
        """
        ssim = self._ssim_region(Ai, Bi, A2i, B2i, ABi, x, y, w, h)

        diff = np.abs(A[y:y+h, x:x+w] - B[y:y+h, x:x+w])
        if diff.size == 0:
            return
        p95 = float(np.percentile(diff, 95))
        p99 = float(np.percentile(diff, 99))
        std = float(diff.std())

        # parar si bloque similar o no hay pico suficiente, o ya es pequeno
        if ssim >= ssim_min or p95 < 0.6 * delta_pix or w <= min_block or h <= min_block:
            if (w <= min_block or h <= min_block) and (ssim < ssim_min) and (p99 >= delta_pix) and (std > 2.0):
                out_rects.append((x, y, w, h, 1.0 - ssim))
            return

        # subdividir
        hw, hh = w // 2, h // 2
        if hw == 0 or hh == 0:
            if (ssim < ssim_min) and (p99 >= delta_pix) and (std > 2.0):
                out_rects.append((x, y, w, h, 1.0 - ssim))
            return

        self._quadtree_ssim(Ai, Bi, A2i, B2i, ABi, A, B, x,     y,     hw, hh, ssim_min, min_block, delta_pix, out_rects)
        self._quadtree_ssim(Ai, Bi, A2i, B2i, ABi, A, B, x+hw,  y,     w-hw, hh, ssim_min, min_block, delta_pix, out_rects)
        self._quadtree_ssim(Ai, Bi, A2i, B2i, ABi, A, B, x,     y+hh,  hw, h-hh, ssim_min, min_block, delta_pix, out_rects)
        self._quadtree_ssim(Ai, Bi, A2i, B2i, ABi, A, B, x+hw,  y+hh,  w-hw, h-hh, ssim_min, min_block, delta_pix, out_rects)

    def _merge_rects(self, rects, max_gap=0):
        """Fusiona rectangulos solo si se SOLAPAN (no solo tocar)."""
        if not rects: return []
        rs = [[x, y, x+w, y+h, s] for (x,y,w,h,s) in rects]
        changed = True
        while changed:
            changed = False
            out = []
            rs.sort()
            while rs:
                x1,y1,x2,y2,s = rs.pop()
                merged = False
                for i,(X1,Y1,X2,Y2,S) in enumerate(out):
                    if not (x2 <= X1 - max_gap or X2 <= x1 - max_gap or
                            y2 <= Y1 - max_gap or Y2 <= y1 - max_gap):
                        out[i] = [min(x1,X1), min(y1,Y1), max(x2,X2), max(y2,Y2), max(s,S)]
                        merged = True; changed = True; break
                if not merged:
                    out.append([x1,y1,x2,y2,s])
            rs = out
        return [(x1,y1,x2-x1,y2-y1,s) for (x1,y1,x2,y2,s) in rs]

    def compare_quadtree_ssim(self):
        try:
            A, B = self._prep_two()
            ssim_min = float(self.ent_ssim.get())
        except Exception:
            ssim_min = 0.97
        ssim_min = max(0.0, min(0.999, ssim_min))

        try:
            min_block = int(self.ent_min.get())
        except Exception:
            min_block = 8
        min_block = max(4, min_block)

        try:
            delta_pix = float(self.ent_dp.get())
        except Exception:
            delta_pix = 18.0

        H, W = A.shape
        Ai, Bi, A2i, B2i, ABi = self._build_integrals(A, B)

        rects = []
        self._quadtree_ssim(Ai, Bi, A2i, B2i, ABi, A, B,
                            0, 0, W, H, ssim_min, min_block, delta_pix, rects)
        rects = self._merge_rects(rects, max_gap=0)
        self.regions = rects

        if not rects:
            messagebox.showinfo("Resultado", "No se detectaron diferencias relevantes.")
            self.clear_regions()
            return

        self.draw_regions(rects, W, H)
        messagebox.showinfo("Resultado",
                            f"Regiones: {len(rects)} (ssim_min={ssim_min:.3f}, "
                            f"min_bloque={min_block}, delta_pix={delta_pix:.1f})")

    # ---------- overlays ----------

    def draw_regions(self, regions, img_w=None, img_h=None):
        if self.disp_a: self.disp_a['canvas'].delete("region")
        if self.disp_b: self.disp_b['canvas'].delete("region")
        if not regions: return
        if img_w is None or img_h is None:
            try:
                A, _ = self._prep_two(); img_h, img_w = A.shape
            except: return
        for (x, y, w, h, score) in regions:
            for disp in (self.disp_a, self.disp_b):
                if not disp: continue
                sx, sy = disp['w']/img_w, disp['h']/img_h
                x1 = int(disp['ox'] + x*sx)
                y1 = int(disp['oy'] + y*sy)
                x2 = int(disp['ox'] + (x+w)*sx)
                y2 = int(disp['oy'] + (y+h)*sy)
                disp['canvas'].create_rectangle(x1, y1, x2, y2,
                                                outline="red", width=2, tag="region")

    def clear_regions(self):
        self.regions = []
        if self.disp_a: self.disp_a['canvas'].delete("region")
        if self.disp_b: self.disp_b['canvas'].delete("region")

if __name__ == "__main__":
    app = ComparadorGUI()
    app.mainloop()
