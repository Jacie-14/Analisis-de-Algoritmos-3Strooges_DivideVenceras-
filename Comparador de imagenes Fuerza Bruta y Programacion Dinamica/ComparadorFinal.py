import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageFilter, Image as PILImage
import numpy as np
from collections import Counter
import heapq
import time


# HERRAMIENTAS NUMÉRICAS

def match_histogram(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    s = src.clip(0, 255).astype(np.uint8).ravel()
    r = ref.clip(0, 255).astype(np.uint8).ravel()

    s_hist = np.bincount(s, minlength=256).astype(np.float64)
    r_hist = np.bincount(r, minlength=256).astype(np.float64)

    s_cdf = np.cumsum(s_hist) / s_hist.sum()
    r_cdf = np.cumsum(r_hist) / r_hist.sum()

    inv = np.interp(s_cdf, r_cdf, np.arange(256))
    out = inv[s].reshape(src.shape)
    return out.astype(np.float32)


def coarse_align_by_shift(A, B, max_shift=8, step=1):
    H, W = A.shape

    def blur(im):
        t = np.pad(im, ((0,0),(1,1)), mode="edge")
        imx = (t[:, :-2] + 2*t[:, 1:-1] + t[:, 2:]) * 0.25
        t = np.pad(imx, ((1,1),(0,0)), mode="edge")
        imy = (t[:-2, :] + 2*t[1:-1, :] + t[2:, :]) * 0.25
        return imy

    Ab = blur(A)
    Bb = blur(B)

    best = (1e18, 0, 0)

    for dy in range(-max_shift, max_shift+1, step):
        y0 = max(0, dy)
        y1 = min(H, H+dy)
        Y0 = -dy if dy < 0 else 0

        for dx in range(-max_shift, max_shift+1, step):
            x0 = max(0, dx)
            x1 = min(W, W+dx)
            X0 = -dx if dx < 0 else 0

            if y1 - y0 < H // 2 or x1 - x0 < W // 2:
                continue

            a = Ab[y0:y1, x0:x1]
            b = Bb[Y0:Y0+(y1-y0), X0:X0+(x1-x0)]

            mse = float(((a - b)**2).mean())
            if mse < best[0]:
                best = (mse, dx, dy)

    _, dx, dy = best

    B_shift = np.zeros_like(B)
    y0 = max(0, dy)
    y1 = min(H, H+dy)
    Y0 = -dy if dy < 0 else 0

    x0 = max(0, dx)
    x1 = min(W, W+dx)
    X0 = -dx if dx < 0 else 0

    B_shift[y0:y1, x0:x1] = B[Y0:Y0+(y1-y0), X0:X0+(x1-x0)]

    return B_shift, dx, dy


def integral_image(a):
    return np.pad(a, ((1,0),(1,0)), mode="constant").cumsum(0).cumsum(1)


def box_sum(ii, x0, y0, x1, y1):
    return float(ii[y1,x1] - ii[y0,x1] - ii[y1,x0] + ii[y0,x0])


# HUFFMAN

class HuffmanNode:
    def __init__(self, value=None, freq=0, left=None, right=None):
        self.value = value
        self.freq = freq
        self.left = left
        self.right = right
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(data):
    freq = Counter(data)
    heap_list = [HuffmanNode(v,f) for v,f in freq.items()]
    heapq.heapify(heap_list)

    while len(heap_list)>1:
        n1 = heapq.heappop(heap_list)
        n2 = heapq.heappop(heap_list)
        merged = HuffmanNode(None, n1.freq+n2.freq, n1, n2)
        heapq.heappush(heap_list, merged)

    return heap_list[0]

def build_codes(node, prefix="", code_map=None):
    if code_map is None:
        code_map = {}

    if node.value is not None:
        code_map[node.value] = prefix
    else:
        build_codes(node.left, prefix+"0", code_map)
        build_codes(node.right, prefix+"1", code_map)

    return code_map

def huffman_encode(data):
    root = build_huffman_tree(data)
    code_map = build_codes(root)
    encoded = "".join(code_map[b] for b in data)
    return encoded, root

def huffman_decode(bits, root):
    decoded=[]
    node=root
    for bit in bits:
        node = node.left if bit=="0" else node.right
        if node.value is not None:
            decoded.append(node.value)
            node=root
    return np.array(decoded, dtype=np.uint8)


# GUI

class ComparadorGUI(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title("Comparador de Imágenes)")
        self.geometry("1200x750")
        self.minsize(1000,600)

        # Memo para Programación Dinámica del Quadtree
        self.dp_ssim={}

        self.img_a=None
        self.img_b=None
        self.tk_a=None
        self.tk_b=None
        self.disp_a=None
        self.disp_b=None

        self.regions=[]

        self._build_menu()
        self._build_layout()


    # MENU

    def _build_menu(self):
        menubar=tk.Menu(self)

        # Archivo
        archivo=tk.Menu(menubar, tearoff=0)
        archivo.add_command(label="Cargar Imagen A", command=lambda:self.load_image("A"))
        archivo.add_command(label="Cargar Imagen B", command=lambda:self.load_image("B"))
        archivo.add_separator()
        archivo.add_command(label="Salir", command=self.destroy)
        menubar.add_cascade(label="Archivo", menu=archivo)

        # Compresión
        comp=tk.Menu(menubar, tearoff=0)
        comp.add_command(label="Comprimir A (Huffman)", command=lambda:self.compress_image(self.img_a))
        comp.add_command(label="Comprimir B (Huffman)", command=lambda:self.compress_image(self.img_b))
        comp.add_command(label="Descomprimir última", command=self.decompress_image)
        menubar.add_cascade(label="Compresión", menu=comp)

        # Comparación
        comparacion=tk.Menu(menubar, tearoff=0)
        comparacion.add_command(label="Comparar (algoritmo seleccionado)", command=self.compare_dispatch)
        menubar.add_cascade(label="Comparación", menu=comparacion)

        self.config(menu=menubar)


    # LAYOUT

    def _build_layout(self):

        container = ttk.Frame(self, padding=10)
        container.pack(fill="both", expand=True)

        left=ttk.LabelFrame(container, text="Imagen A", padding=10)
        left.pack(side="left", fill="both", expand=True)
        self.canvas_a=tk.Canvas(left, bg="#222")
        self.canvas_a.pack(fill="both", expand=True)

        right=ttk.LabelFrame(container, text="Imagen B", padding=10)
        right.pack(side="left", fill="both", expand=True)
        self.canvas_b=tk.Canvas(right, bg="#222")
        self.canvas_b.pack(fill="both", expand=True)

        bottom=ttk.Frame(self, padding=10)
        bottom.pack(fill="x", side="bottom")

        params=ttk.Frame(bottom)
        params.pack(side="left")

        ttk.Label(params, text="ssim_min").grid(row=0,column=0)
        self.ent_ssim=ttk.Entry(params, width=6)
        self.ent_ssim.insert(0,"0.97")
        self.ent_ssim.grid(row=0,column=1, padx=5)

        ttk.Label(params, text="min_bloque").grid(row=0,column=2)
        self.ent_min=ttk.Entry(params, width=6)
        self.ent_min.insert(0,"8")
        self.ent_min.grid(row=0,column=3, padx=5)

        ttk.Label(params, text="hp_radius").grid(row=0,column=4)
        self.ent_hp=ttk.Entry(params, width=4)
        self.ent_hp.insert(0,"8")
        self.ent_hp.grid(row=0,column=5, padx=5)

        ttk.Label(params, text="delta_pix").grid(row=0,column=6)
        self.ent_dp=ttk.Entry(params, width=6)
        self.ent_dp.insert(0,"18")
        self.ent_dp.grid(row=0,column=7, padx=5)

        self.modo_robusto=tk.BooleanVar(value=True)
        ttk.Checkbutton(params, text="Robusto luz", variable=self.modo_robusto)\
            .grid(row=0,column=8, padx=5)

        # Selector de algoritmo
        ttk.Label(params, text="Algoritmo").grid(row=1, column=0, pady=(6,0), sticky="w")
        self.algoritmo = tk.StringVar(value="Quadtree+PD")
        self.cbo_algo = ttk.Combobox(
            params,
            textvariable=self.algoritmo,
            state="readonly",
            width=18,
            values=["Fuerza bruta", "Quadtree+PD"]
        )
        self.cbo_algo.grid(row=1, column=1, columnspan=3, pady=(6,0), sticky="w")

        self.btn_clear=ttk.Button(bottom, text="Limpiar", state="disabled",
                                  command=self.clear_regions)
        self.btn_clear.pack(side="right", padx=6)

        self.btn_compare=ttk.Button(bottom, text="Comparar (algoritmo seleccionado)",
                                   state="disabled",
                                   command=self.compare_dispatch)
        self.btn_compare.pack(side="right", padx=6)

        self.bind("<Configure>", self._on_resize)


    # FUNC HUFFMAN

    def compress_image(self, pil_img):
        if pil_img is None:
            messagebox.showerror("Error","Debes cargar la imagen.")
            return

        gray=pil_img.convert("L")
        arr=np.asarray(gray, dtype=np.uint8).ravel()
        encoded,tree=huffman_encode(arr)
        ratio=len(encoded)/(len(arr)*8)

        self.last_huffman=(encoded,tree,gray.size)
        messagebox.showinfo("Huffman",f"Ratio: {ratio:.3f}")

    def decompress_image(self):
        if not hasattr(self,"last_huffman"):
            messagebox.showwarning("Aviso","No has comprimido nada.")
            return

        encoded,tree,size=self.last_huffman
        decoded=huffman_decode(encoded, tree)
        arr=decoded.reshape(size[::-1])
        PILImage.fromarray(arr,"L").show()


    # CARGA DE IMÁGENES

    def load_image(self, side):

        path=filedialog.askopenfilename(
            filetypes=[("Imágenes","*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]
        )
        if not path:
            return

        try:
            img=PILImage.open(path).convert("RGB")
        except:
            messagebox.showerror("Error","No se pudo abrir la imagen")
            return

        if side=="A":
            self.img_a=img
            self._display(self.canvas_a, img, "A")
        else:
            self.img_b=img
            self._display(self.canvas_b, img, "B")

        if self.img_a and self.img_b:
            self.btn_compare.config(state="normal")
            self.btn_clear.config(state="normal")


    def _display(self, canvas, img, which):
        cw,ch=canvas.winfo_width(),canvas.winfo_height()

        if cw<5 or ch<5:
            self.update_idletasks()
            cw,ch=canvas.winfo_width(),canvas.winfo_height()

        img2=img.copy()
        img2.thumbnail((cw,ch))

        tkimg=ImageTk.PhotoImage(img2)
        canvas.delete("all")
        canvas.create_image(cw//2, ch//2, image=tkimg)

        disp_w,disp_h=img2.size
        ox,oy=(cw-disp_w)//2,(ch-disp_h)//2

        if which=="A":
            self.tk_a=tkimg
            self.disp_a={"w":disp_w,"h":disp_h,"ox":ox,"oy":oy,"canvas":canvas}
        else:
            self.tk_b=tkimg
            self.disp_b={"w":disp_w,"h":disp_h,"ox":ox,"oy":oy,"canvas":canvas}


    def _on_resize(self,_):
        if self.img_a:
            self._display(self.canvas_a, self.img_a,"A")
        if self.img_b:
            self._display(self.canvas_b, self.img_b,"B")
        if self.regions:
            self.draw_regions(self.regions)


    # PREPARATIVOS

    def _to_highpass(self, pil_gray, radius):
        base=np.asarray(pil_gray, np.float32)
        blur=np.asarray(pil_gray.filter(ImageFilter.GaussianBlur(radius)), np.float32)
        hp=base-blur
        hp-=hp.min()
        m=hp.max()
        if m>0: hp*=255.0/m
        return hp.astype(np.float32)


    def _prep_two(self):
        """
        Prepara dos imágenes para el método Quadtree+PD:
        - Escala a tamaño común
        - Igualación de histograma
        - Alineación por traslación
        - (Opcional) filtro paso alto
        """
        if self.img_a is None or self.img_b is None:
            raise ValueError("Faltan imágenes.")

        a=self.img_a.convert("L")
        b=self.img_b.convert("L")

        w=min(a.width,b.width)
        h=min(a.height,b.height)

        a=a.resize((w,h))
        b=b.resize((w,h))

        A=np.asarray(a, np.float32)
        B=np.asarray(b, np.float32)

        B=match_histogram(B,A)
        B,_,_=coarse_align_by_shift(A,B)

        hp_r=max(4, int(self.ent_hp.get()))

        if self.modo_robusto.get():
            A=self._to_highpass(PILImage.fromarray(A.astype(np.uint8)), hp_r)
            B=self._to_highpass(PILImage.fromarray(B.astype(np.uint8)), hp_r)

        return A,B

    def _prep_two_simple(self):
        """
        Prepara dos imágenes para FUERZA BRUTA:
        - Solo escala a tamaño común y pasa a escala de grises.
        - Sin igualación de histograma ni filtros, para comparar
          más "literalmente" los píxeles.
        """
        if self.img_a is None or self.img_b is None:
            raise ValueError("Faltan imágenes.")

        a=self.img_a.convert("L")
        b=self.img_b.convert("L")

        w=min(a.width,b.width)
        h=min(a.height,b.height)

        a=a.resize((w,h))
        b=b.resize((w,h))

        A=np.asarray(a, np.int16)
        B=np.asarray(b, np.int16)

        return A,B


    # SSIM REGION

    def _build_integrals(self, A,B):
        Ai=integral_image(A)
        Bi=integral_image(B)
        A2i=integral_image(A*A)
        B2i=integral_image(B*B)
        ABi=integral_image(A*B)
        return Ai,Bi,A2i,B2i,ABi


    def _ssim_region(self, Ai,Bi,A2i,B2i,ABi, x,y,w,h):
        area=w*h
        if area<=0:
            return 1.0

        sa=box_sum(Ai,x,y,x+w,y+h); ma=sa/area
        sb=box_sum(Bi,x,y,x+w,y+h); mb=sb/area

        sa2=box_sum(A2i,x,y,x+w,y+h); va=max(sa2/area - ma*ma,1e-6)
        sb2=box_sum(B2i,x,y,x+w,y+h); vb=max(sb2/area - mb*mb,1e-6)

        sab=box_sum(ABi,x,y,x+w,y+h)
        cov=sab/area - ma*mb

        L=255
        C1=(0.01*L)**2
        C2=(0.03*L)**2

        num=(2*ma*mb + C1)*(2*cov + C2)
        den=(ma*ma+mb*mb+C1)*(va+vb+C2)

        return float(max(0,min(1,num/den)))


    # QUAD TREE + PD

    def _quadtree_ssim(self, Ai,Bi,A2i,B2i,ABi, A,B,
                       x,y,w,h, ssim_min, min_block, delta_pix, out_rects):
        """
        Divide y vencerás + Programación Dinámica (memo) + poda:

        - Divide recursivamente en 4 subregiones si la similitud no es suficiente.
        - Usa dp_ssim como cache (memoización) para no recalcular mismas regiones.
        - Aplica poda: si la región es suficientemente similar o no tiene
          diferencia "fuerte", no sigue explorando (backtracking).
        """
        key=(x,y,w,h)

        if key in self.dp_ssim:
            ssim,p95,p99,std = self.dp_ssim[key]
        else:
            # calcular SSIM
            ssim=self._ssim_region(Ai,Bi,A2i,B2i,ABi, x,y,w,h)

            diff=np.abs(A[y:y+h, x:x+w] - B[y:y+h, x:x+w])
            if diff.size==0:
                return

            p95=float(np.percentile(diff,95))
            p99=float(np.percentile(diff,99))
            std=float(diff.std())

            self.dp_ssim[key]=(ssim,p95,p99,std)

        # ---- Poda: condiciones de parada ----
        if ssim>=ssim_min or p95<0.6*delta_pix or w<=min_block or h<=min_block:
            # Caso base: hoja "mala" (diferente) que vale la pena marcar
            if (w<=min_block or h<=min_block) and ssim<ssim_min and p99>=delta_pix and std>2.0:
                out_rects.append((x,y,w,h,1.0-ssim))
            # backtracking: no subdividimos más esta rama
            return

        # ---- Divide (divide y vencerás) ----
        hw=w//2
        hh=h//2

        if hw==0 or hh==0:
            if ssim<ssim_min and p99>=delta_pix and std>2.0:
                out_rects.append((x,y,w,h,1.0-ssim))
            return

        # backtracking + recursión: explora 4 subregiones
        self._quadtree_ssim(Ai,Bi,A2i,B2i,ABi, A,B, x,    y,    hw,hh, ssim_min,min_block,delta_pix,out_rects)
        self._quadtree_ssim(Ai,Bi,A2i,B2i,ABi, A,B, x+hw, y,    w-hw,hh, ssim_min,min_block,delta_pix,out_rects)
        self._quadtree_ssim(Ai,Bi,A2i,B2i,ABi, A,B, x,    y+hh, hw,h-hh, ssim_min,min_block,delta_pix,out_rects)
        self._quadtree_ssim(Ai,Bi,A2i,B2i,ABi, A,B, x+hw,y+hh, w-hw,h-hh, ssim_min,min_block,delta_pix,out_rects)


    def _merge_rects(self, rects, max_gap=0):
        if not rects:
            return []

        rs=[[x,y,x+w,y+h,s] for (x,y,w,h,s) in rects]
        changed=True

        while changed:
            changed=False
            out=[]
            rs.sort()

            while rs:
                x1,y1,x2,y2,s=rs.pop()
                merged=False

                for i,(X1,Y1,X2,Y2,S) in enumerate(out):
                    if not (x2<=X1-max_gap or X2<=x1-max_gap or y2<=Y1-max_gap or Y2<=y1-max_gap):
                        out[i]=[
                            min(x1,X1),min(y1,Y1),
                            max(x2,X2),max(y2,Y2),
                            max(s,S)
                        ]
                        merged=True
                        changed=True
                        break

                if not merged:
                    out.append([x1,y1,x2,y2,s])

            rs=out

        return [(x1,y1,x2-x1,y2-y1,s) for (x1,y1,x2,y2,s) in rs]


    # METODO QUADTREE + PD

    def compare_quadtree_ssim(self):
        self.dp_ssim.clear()

        t0 = time.time()
        A,B=self._prep_two()

        try:
            ssim_min=float(self.ent_ssim.get())
        except: ssim_min=0.97

        try:
            min_block=int(self.ent_min.get())
        except: min_block=8

        try:
            delta_pix=float(self.ent_dp.get())
        except: delta_pix=18.0

        H,W=A.shape
        Ai,Bi,A2i,B2i,ABi=self._build_integrals(A,B)

        rects=[]
        self._quadtree_ssim(Ai,Bi,A2i,B2i,ABi,A,B,
                            0,0,W,H, ssim_min,min_block,delta_pix,
                            rects)

        rects=self._merge_rects(rects)
        self.regions=rects

        dt = time.time() - t0

        if not rects:
            messagebox.showinfo("Resultado",
                f"No se detectaron diferencias importantes.\n"
                f"Tiempo: {dt:.3f} s\n"
                f"Método: Quadtree + Programación Dinámica")
            self.clear_regions()
            return

        self.draw_regions(rects,W,H)

        messagebox.showinfo("Resultado",
            f"Rectángulos detectados: {len(rects)}\n"
            f"Tiempo: {dt:.3f} s\n"
            "Método: Quadtree + Programación Dinámica (divide y vencerás + memo + poda)")


    # FUERZA BRUTA

    def compare_bruteforce(self):
        if self.img_a is None or self.img_b is None:
            messagebox.showerror("Error","Carga ambas imágenes.")
            return

        t0 = time.time()
        A,B = self._prep_two_simple()

        h, w = A.shape
        diff = np.abs(A - B)

        # Umbral para considerar "pixel diferente"
        thr = 5.0
        different = (diff > thr)
        total_pixels = w * h
        num_diff = int(different.sum())
        num_equal = total_pixels - num_diff
        percent_equal = 100.0 * num_equal / max(1, total_pixels)

        # Para visualizar: marcamos bloques (ej. 16x16) que tengan alguna diferencia
        block = 16
        rects = []
        for y in range(0, h, block):
            for x in range(0, w, block):
                sub = different[y:y+block, x:x+block]
                if sub.size > 0 and sub.any():
                    hh, ww = sub.shape
                    rects.append((x, y, ww, hh, 1.0))

        self.regions = rects
        self.draw_regions(rects, w, h)

        dt = time.time() - t0

        messagebox.showinfo(
            "Fuerza Bruta",
            f"Pixeles distintos: {num_diff} / {total_pixels}\n"
            f"Porcentaje de igualdad: {percent_equal:.2f}%\n"
            f"Bloques marcados: {len(rects)}\n"
            f"Tiempo: {dt:.3f} s\n"
            "Método: Fuerza bruta (comparación pixel a pixel)"
        )


    # SELECCIÓN DE ALGORITMO

    def compare_dispatch(self):
        algo = self.algoritmo.get()
        if algo == "Fuerza bruta":
            self.compare_bruteforce()
        else:
            self.compare_quadtree_ssim()


    # DIBUJAR RECTÁNGULOS

    def draw_regions(self, regions, img_w, img_h):
        if self.disp_a:
            self.disp_a['canvas'].delete("region")
        if self.disp_b:
            self.disp_b['canvas'].delete("region")

        for (x,y,w,h,s) in regions:
            for disp in (self.disp_a, self.disp_b):
                if not disp:
                    continue
                sx=disp['w']/img_w
                sy=disp['h']/img_h

                x1=int(disp['ox'] + x*sx)
                y1=int(disp['oy'] + y*sy)
                x2=int(disp['ox'] + (x+w)*sx)
                y2=int(disp['oy'] + (y+h)*sy)

                disp['canvas'].create_rectangle(
                    x1,y1,x2,y2,
                    outline="red", width=2, tag="region"
                )


    def clear_regions(self):
        self.regions=[]
        if self.disp_a:
            self.disp_a['canvas'].delete("region")
        if self.disp_b:
            self.disp_b['canvas'].delete("region")


# MAINNN

if __name__=="__main__":
    app=ComparadorGUI()
    app.mainloop()
