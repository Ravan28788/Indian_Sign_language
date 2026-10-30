
import os
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from io import BytesIO


class NPYToPNGConverter:
    def __init__(self, root):
        self.root = root
        self.root.title("NPY to PNG Converter")
        self.root.geometry("750x600")

        self.array = None
        self.preview = None
        self.png_image = None

        tk.Label(
            root,
            text="NPY to PNG Converter",
            font=("Arial", 20, "bold")
        ).pack(pady=15)

        tk.Button(
            root,
            text="Open NPY File",
            command=self.open_file,
            width=22
        ).pack(pady=8)

        self.file_label = tk.Label(
            root,
            text="No file selected",
            wraplength=700
        )
        self.file_label.pack(pady=5)

        self.info_label = tk.Label(
            root,
            text="Array information will appear here"
        )
        self.info_label.pack(pady=5)

        self.preview_label = tk.Label(
            root,
            text="Image Preview",
            width=70,
            height=20,
            relief="groove"
        )
        self.preview_label.pack(
            padx=20, pady=15, expand=True
        )

        tk.Button(
            root,
            text="Save As PNG",
            command=self.save_file,
            width=22
        ).pack(pady=15)

    def convert_array(self, array):
        array = np.asarray(array)

        if array.size == 0:
            raise ValueError("The selected array is empty.")

        if not np.issubdtype(array.dtype, np.number):
            raise ValueError("The array must contain numbers.")

        if not np.isfinite(array).all():
            raise ValueError(
                "The array contains NaN or infinite values."
            )

        if array.ndim == 1:
            fig = Figure(figsize=(8, 4), dpi=100)
            ax = fig.add_subplot(111)

            ax.plot(array, linewidth=1.5)
            ax.set_title("NPY Array Visualization")
            ax.set_xlabel("Index")
            ax.set_ylabel("Value")
            ax.grid(True)

            canvas = FigureCanvasAgg(fig)
            canvas.draw()

            buffer = BytesIO()
            canvas.print_png(buffer)
            buffer.seek(0)

            image = Image.open(buffer).convert("RGB").copy()
            buffer.close()
            return image

        if array.ndim == 2:
            if np.issubdtype(array.dtype, np.floating):
                minimum = array.min()
                maximum = array.max()

                if minimum >= 0 and maximum <= 1:
                    array = array * 255
                elif minimum < 0 or maximum > 255:
                    if maximum == minimum:
                        array = np.zeros_like(array)
                    else:
                        array = (
                            (array - minimum)
                            / (maximum - minimum)
                            * 255
                        )

            elif array.dtype == np.uint16:
                array = array / 257

            array = np.clip(array, 0, 255).astype(np.uint8)
            return Image.fromarray(array)

        if array.ndim == 3:
            if array.shape[2] not in (1, 3, 4):
                raise ValueError(
                    "Unsupported channel layout. "
                    "Expected grayscale, RGB, or RGBA."
                )

            if array.shape[2] == 1:
                array = array[:, :, 0]

                if np.issubdtype(array.dtype, np.floating):
                    minimum = array.min()
                    maximum = array.max()

                    if minimum >= 0 and maximum <= 1:
                        array = array * 255
                    elif minimum < 0 or maximum > 255:
                        if maximum == minimum:
                            array = np.zeros_like(array)
                        else:
                            array = (
                                (array - minimum)
                                / (maximum - minimum)
                                * 255
                            )

                array = np.clip(
                    array, 0, 255
                ).astype(np.uint8)

                return Image.fromarray(array)

            if np.issubdtype(array.dtype, np.floating):
                minimum = array.min()
                maximum = array.max()

                if minimum >= 0 and maximum <= 1:
                    array = array * 255
                elif minimum < 0 or maximum > 255:
                    if maximum == minimum:
                        array = np.zeros_like(array)
                    else:
                        array = (
                            (array - minimum)
                            / (maximum - minimum)
                            * 255
                        )

            array = np.clip(array, 0, 255).astype(np.uint8)
            return Image.fromarray(array)

        raise ValueError(
            f"Unsupported array shape: {array.shape}. "
            "This program supports 1D data, 2D images, "
            "and 3D grayscale/RGB/RGBA images."
        )

    def open_file(self):
        path = filedialog.askopenfilename(
            title="Select NPY File",
            filetypes=[
                ("NumPy files", "*.npy"),
                ("All files", "*.*")
            ]
        )

        if not path:
            return

        try:
            array = np.load(path, allow_pickle=False)
            image = self.convert_array(array)

            self.array = array
            self.png_image = image

            self.file_label.config(
                text=f"Selected file: {os.path.basename(path)}"
            )

            self.info_label.config(
                text=(
                    f"Shape: {array.shape} | "
                    f"Type: {array.dtype} | "
                    f"Dimensions: {array.ndim}"
                )
            )

            preview = image.copy()
            preview.thumbnail((550, 330))

            self.preview = ImageTk.PhotoImage(preview)

            self.preview_label.config(
                image=self.preview,
                text=""
            )

        except Exception as error:
            messagebox.showerror(
                "File Error",
                str(error)
            )

    def save_file(self):
        if self.png_image is None:
            messagebox.showwarning(
                "No File",
                "Please open an NPY file first."
            )
            return

        path = filedialog.asksaveasfilename(
            title="Save As PNG",
            defaultextension=".png",
            initialfile="converted_image.png",
            filetypes=[("PNG Image", "*.png")]
        )

        if not path:
            return

        try:
            self.png_image.save(path, format="PNG")

            messagebox.showinfo(
                "Success",
                f"PNG saved successfully:\n{path}"
            )

        except Exception as error:
            messagebox.showerror(
                "Save Error",
                str(error)
            )


if __name__ == "__main__":
    root = tk.Tk()
    app = NPYToPNGConverter(root)
    root.mainloop()