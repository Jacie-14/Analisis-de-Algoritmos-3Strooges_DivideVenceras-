import tkinter as tk
from tkinter import ttk, messagebox
import random
import time 
from itertools import combinations

def subset_sum_bruteforce(arr, target):
    """Devuelve TODOS los subconjuntos."""
    sols = []
    iteraciones = 0
    for r in range(len(arr) + 1):
        for combo in combinations(arr, r):
            iteraciones += 1
            if sum(combo) == target:
                sols.append(list(combo))
    return sols, iteraciones  # <-- Ahora regresamos también iteraciones

class SubsetSumApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Suma de Subconjuntos - Fuerza Bruta")
        self.var_N = tk.StringVar(value="8")
        self.var_min = tk.StringVar(value="0")
        self.var_max = tk.StringVar(value="20")
        self.var_lista = tk.StringVar(value="")
        self.var_target = tk.StringVar(value="10")
        self._build_ui()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")

        # Parámetros de generación
        frm = ttk.LabelFrame(main, text="Parámetros", padding=5)
        frm.grid(row=0, column=0, sticky="ew")
        ttk.Label(frm, text="N:").grid(row=0, column=0)
        ttk.Entry(frm, textvariable=self.var_N, width=5).grid(row=0, column=1)
        ttk.Label(frm, text="Min:").grid(row=0, column=2)
        ttk.Entry(frm, textvariable=self.var_min, width=5).grid(row=0, column=3)
        ttk.Label(frm, text="Max:").grid(row=0, column=4)
        ttk.Entry(frm, textvariable=self.var_max, width=5).grid(row=0, column=5)
        ttk.Button(frm, text="Generar", command=self.generar_lista_aleatoria).grid(row=0, column=6, padx=5)

        # Lista
        ttk.Label(main, text="Lista:").grid(row=1, column=0, sticky="w")
        ttk.Entry(main, textvariable=self.var_lista).grid(row=2, column=0, sticky="ew", pady=3)

        # Objetivo
        frm2 = ttk.Frame(main)
        frm2.grid(row=3, column=0, sticky="ew")
        ttk.Label(frm2, text="Suma objetivo T:").grid(row=0, column=0)
        ttk.Entry(frm2, textvariable=self.var_target, width=10).grid(row=0, column=1, padx=5)
        ttk.Button(frm2, text="Resolver", command=self.resolver).grid(row=0, column=2, padx=5)

        # Resultados
        self.txt_result = tk.Text(main, height=10, wrap="word")
        self.txt_result.grid(row=4, column=0, sticky="nsew", pady=5)

    def generar_lista_aleatoria(self):
        try:
            n = int(self.var_N.get()); a = int(self.var_min.get()); b = int(self.var_max.get())
            if n < 0 or a > b: raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Verifica N, min y max")
            return
        arr = [random.randint(a, b) for _ in range(n)]
        self.var_lista.set(", ".join(map(str, arr)))

    def resolver(self):
        self.txt_result.delete("1.0", tk.END)
        try:
            arr = [float(x) if "." in x else int(x) for x in self.var_lista.get().replace(",", " ").split()]
            target = float(self.var_target.get())
            if not arr: raise ValueError
        except:
            messagebox.showerror("Error", "Revisa la lista y el objetivo")
            return

        # --- Medición de tiempo ---
        inicio = time.perf_counter()
        soluciones, iteraciones = subset_sum_bruteforce(arr, target)
        fin = time.perf_counter()
        tiempo_total = fin - inicio

        # --- Mostrar resultados ---
        self.txt_result.insert(tk.END, f"Iteraciones totales: {iteraciones}\n")
        self.txt_result.insert(tk.END, f"Tiempo de cálculo: {tiempo_total:.6f} segundos\n\n")
        if not soluciones:
            self.txt_result.insert(tk.END, "No se encontraron soluciones.\n")
        else:
            self.txt_result.insert(tk.END, f"Se encontraron {len(soluciones)} soluciones:\n\n")
            for i, s in enumerate(soluciones, 1):
                self.txt_result.insert(tk.END, f"{i}) {s}  → suma={sum(s)}\n")

if __name__ == "__main__":
    root = tk.Tk()
    root.minsize(500, 350)
    SubsetSumApp(root)
    root.mainloop()
