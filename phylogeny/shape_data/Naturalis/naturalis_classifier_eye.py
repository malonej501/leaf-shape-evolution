"""This script displays herbarium images in order and allows classification
by saving keystrokes in a file and switching to the next image. Overwrite the
previous label by pressing [r]. Leaves are classified as u (unlobed), l
(lobed), d (dissected), c (compound) or a (ambiguous)."""
import os
import re
import tkinter as tk
import io
import traceback
import pandas as pd
import requests
from pynput.keyboard import Listener
from PIL import Image, ImageTk
from google_images_search import GoogleImagesSearch
from duckduckgo_search import DDGS
from selenium import webdriver

# wd = "sample_eud_21-1-24/"
# WD = "jan_zun_nat_ang_26-09-24/"
WD = "jan_zun_nat_ang_11-07-25/"
# file = "Naturalis_eud_sample_Janssens_intersect_21-01-24.csv"
SP_FULL = WD + "jan_zun_union_nat_species_11-07-25.csv"
IMG_DIR = WD + "jan_zun_union_nat_species_11-07-25"
DDGI_DIR = WD + "ddgi_sample_" + SP_FULL.split(".")[0]

STARTFROM = 5014  # which leaf index to start labelling from

# Google search engine initialisation
# API key for leaf-project tied to malonej501@gmail.com
API_KEY = "REMOVED FOR SECURITY REASONS - SEE LOCAL MACHINE"
# search engine id for leaf-finder within leaf-project tied to
# malonej501@gmail.com
CX = "25948abc932774194"


def check_alldownloaded():
    """Check if all images from the sample have been downloaded."""
    sp_full_df = pd.read_csv(SP_FULL)
    fnames_exp = [
        f"{value}{index}.png" for index, value
        in enumerate(sp_full_df["species"])
    ]
    fnames = os.listdir(IMG_DIR)

    missing_imgs = list(set(fnames_exp).difference(fnames))

    print("Missing imgs:")
    if missing_imgs:
        for i in missing_imgs:
            print(f"missing {i}")
        print("ERROR: not all images in sample are present")
        print(f"Missing {len(missing_imgs)} images")
    else:
        print("None!")


def search_google_images(query, canvas, root):
    """Search Google Images for a query and display results in a Tkinter
    canvas."""
    print(f"Query: {query}")
    gis = GoogleImagesSearch(API_KEY, CX, validate_images=False)

    _search_params = {
        "q": query,
        "num": 10,
        # "fileType": "jpg|gif|png",
        "rights": "cc_publicdomain|cc_attribute|cc_sharealike|" +
        "cc_noncommercial|cc_nonderived",
        "safe": "safeUndefined",
        "imgType": "imgTypeUndefined",
        "imgSize": "imgSizeUndefined",
        "imgDominantColor": "imgDominantColorUndefined",
        "imgColorType": "imgColorTypeUndefined",
    }

    gis.search(search_params=_search_params)

    for index, item in enumerate(gis.results(), start=1):
        link = item.url
        print(f"result {index} {link}")

        try:
            # Attempt to open the image
            response = requests.get(
                link, stream=True, headers={"User-Agent": "Leaf-Search/1.0"},
                timeout=10
            )
            response.raise_for_status()

            # Check if the content is a valid image
            image = Image.open(io.BytesIO(response.content))
            image.thumbnail((250, 250))

            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)

            # Create label with image
            label = tk.Label(
                canvas, text=f"Result {index}", compound=tk.TOP, image=photo
            )
            label.image = photo
            label.grid(row=(index + 1) %
                       2, column=(index + 1) // 2, sticky="nw")
        except Exception as e:
            print(f"Error processing image {index}: {e}")
    print("Search complete!")


def search_wikimedia_commons(query, canvas):
    """Search Wikimedia Commons for images related to a query and display
    results in a Tkinter canvas."""

    base_url = "https://commons.wikimedia.org/w/api.php"

    # Initialize the Firefox browser in headless mode
    driver_path = (
        "/home/jmalone/Documents/Leaf-Project/Software/Firefox-geckodriver/" +
        "geckodriver"
    )
    os.environ["PATH"] = driver_path
    options = webdriver.FirefoxOptions()
    options.add_argument("--headless")
    driver = webdriver.Firefox(options=options)

    # params = {
    #     "action": "query",
    #     "format": "json",
    #     "list": "search",
    #     "srsearch": "Arabidopsis_thaliana",
    #     "srnamespace": "6",  # Namespace for File
    # }
    params = {
        "action": "query",
        "format": "json",
        "list": "allimages",
        "aifrom": query,
        "ailimit": 10,
    }

    response_wiki = requests.get(base_url, params, timeout=10)
    response_wiki_data = response_wiki.json()

    # extract image urls from json response
    urls = [
        image["url"]
        for image in response_wiki_data.get("query", {}).get("allimages", [])
    ]

    for index, item in enumerate(urls):
        url = item
        print(f"result {index} {url}")

        # Visit url with selenium to pre-load the image
        driver.get(url)

        # # Determine the expected content type based on the file extension in the URL
        # file_extension = url.split(".")[-1].lower()
        # expected_content_type = mimetypes.guess_type(f"image.{file_extension}")[0]
        # headers = {"Content-Type": expected_content_type}
        # # print(headers)

        response_img = requests.get(url, timeout=10)

        try:
            # Check if the content is a valid image
            # print(response_img.headers.get("Content-Type", ""))
            image = Image.open(io.BytesIO(response_img.content))
            image.thumbnail((250, 250))

            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)

            # Create label with image
            label = tk.Label(
                canvas, text=f"Result {index}", compound=tk.TOP, image=photo
            )
            label.image = photo
            label.grid(row=(index + 2) %
                       2, column=(index + 2) // 2, sticky="nw")
        except Exception as e:
            print(f"Error processing image {index}: {e}")
    print("Search complete!")


def search_duckduckgo_images(query, canvas):
    """Search DuckDuckGo Images for a query and display results in a Tkinter
    canvas."""
    results = []
    with DDGS() as ddgs:
        keywords = query
        ddgs_images_gen = ddgs.images(keywords, max_results=10)
        for r in ddgs_images_gen:
            results.append(r)

    for index, item in enumerate(results):
        link = item["thumbnail"]
        print(f"result {index} {link}")

        try:
            # Attempt to open the image
            response = requests.get(link, timeout=10)
            # print(response.headers.get("Content-Type", ""))

            if response.status_code == 200:
                # Check if the content is a valid image
                image = Image.open(io.BytesIO(response.content))
                image.thumbnail((250, 250))

                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(image)

                # Create label with image
                label = tk.Label(
                    canvas,
                    text=f"Result {index}",
                    compound=tk.TOP,
                    image=photo,
                )
                label.image = photo
                label.grid(row=(index + 2) %
                           2, column=(index + 2) // 2, sticky="nw")
            else:
                print(
                    f"Failed to download image {index}. Status code: " +
                    f"{response.status_code}"
                )
        except Exception:
            traceback.print_exc()
    print("Search complete!")


def download_duckduckgo_images(query):
    """Download images from DuckDuckGo for a given query and save them to a
    specified directory."""

    if not os.path.exists(WD + f"ddgi_sample_1/{query}"):
        os.mkdir(WD + f"ddgi_sample_1/{query}")
        saved_data = []
        results = []
        with DDGS() as ddgs:
            keywords = query
            ddgs_images_gen = ddgs.images(keywords, max_results=10)
            for r in ddgs_images_gen:
                results.append(r)

        for index, item in enumerate(results):
            saved_data.append(item)
            link = item["thumbnail"]
            print(f"result {index} {link}")

            try:
                # Attempt to open the image
                response = requests.get(link, timeout=10)
                # print(response.headers.get("Content-Type", ""))

                if response.status_code == 200:
                    # Check if the content is a valid image
                    image = Image.open(io.BytesIO(response.content))
                    image.thumbnail((250, 250))

                    image.save(
                        WD + f"ddgi_sample_1/{query}/result_{index}.png")
                else:
                    print(
                        f"Failed to download image {index}. Status code: " +
                        f"{response.status_code}"
                    )
            except Exception:
                traceback.print_exc()
        print("Search complete!")
        saved_data_df = pd.DataFrame(saved_data)
        saved_data_df.to_csv(
            WD + f"ddgi_sample_1/{query}/results_{query}.csv", index=False
        )


def display_downloaded(query, canvas):
    """Display downloaded images from DuckDuckGo in a Tkinter canvas."""
    # path = WD + "ddgi_sample/" + query
    path = os.path.join(DDGI_DIR, query)
    pngs = sorted([file for file in os.listdir(path)
                   if file.lower().endswith(".png")])
    for index, filename in enumerate(pngs):
        png = os.path.join(path, filename)

        try:
            # Check if the content is a valid image
            image = Image.open(png)

            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)

            # Create label with image
            label = tk.Label(
                canvas,
                text=filename,
                compound=tk.TOP,
                image=photo,
            )
            label.image = photo
            label.grid(row=(index + 2) %
                       2, column=(index + 2) // 2, sticky="nw")

        except Exception:
            traceback.print_exc()
    print("Search complete!")


class ImageLabeler:
    """Class to label images interactively using Tkinter."""

    def __init__(self, img_dir, startfrom=0):
        self.img_dir = img_dir
        self.sorted_file_list = self.get_sorted_file_list()
        self.index = startfrom
        self.leafdata = []
        self.current_key = None
        self.redo = False  # Flag to indicate if redo is needed
        self.zoom = 0.6
        self.offset_x = 0
        self.offset_y = 0

    def get_sorted_file_list(self):
        """Get a sorted list of image files in the directory."""
        files = [f for f in os.listdir(self.img_dir) if f.endswith(
            '.png') and re.search(r'\d+', f)]
        return sorted(files, key=lambda x: int(re.search(r'\d+', x).group()))

    def on_key(self, event, root):
        """Handle key press events for labeling images."""
        if event.char == "q":
            root.destroy()
        elif event.char == "r":
            self.redo = True
            root.destroy()
        else:
            self.current_key = event.char
            root.destroy()

    def label_image(self, filename):
        """Display an image and allow the user to label it."""
        imgpath = os.path.join(self.img_dir, filename)
        query = re.sub(r"\d+", "", filename.rstrip(".png"))

        root = tk.Tk()
        root.title(f"Image Viewer: {query}")
        root.geometry("1000x1100")

        canvas_width, canvas_height = 1000, 1100
        canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
        canvas.grid(row=0, column=0, rowspan=6, columnspan=2)

        pil_img = Image.open(imgpath)
        img_w, img_h = pil_img.size
        self.zoom = 0.6
        self.offset_x = 0
        self.offset_y = 0

        def show_img():
            """Display the image on the canvas with current zoom and offset."""
            # Calculate the region to crop
            crop_w = int(canvas_width / self.zoom)
            crop_h = int(canvas_height / self.zoom)
            left = min(max(self.offset_x, 0), img_w - crop_w)
            top = min(max(self.offset_y, 0), img_h - crop_h)
            right = left + crop_w
            bottom = top + crop_h
            cropped = pil_img.crop((left, top, right, bottom))
            img_resized = cropped.resize(
                (canvas_width, canvas_height), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img_resized)
            canvas.photo = photo  # Keep reference
            canvas.delete("all")
            canvas.create_image(0, 0, anchor="nw", image=photo)

        def zoom(event, zoom_in=True):
            """Zoom in or out based on mouse wheel event."""
            # Mouse position on canvas
            mouse_x = event.x
            mouse_y = event.y
            # Position in image coordinates
            img_x = self.offset_x + mouse_x / self.zoom
            img_y = self.offset_y + mouse_y / self.zoom
            # Update zoom
            old_zoom = self.zoom
            if zoom_in:
                self.zoom *= 1.1
            else:
                self.zoom /= 1.1
            # Prevent too much zoom out
            self.zoom = max(self.zoom, 0.6)
            # Adjust offset so that the point under the mouse stays fixed
            self.offset_x = img_x - mouse_x / self.zoom
            self.offset_y = img_y - mouse_y / self.zoom
            # Clamp offset
            self.offset_x = min(max(self.offset_x, 0),
                                img_w - canvas_width / self.zoom)
            self.offset_y = min(max(self.offset_y, 0),
                                img_h - canvas_height / self.zoom)
            show_img()

        show_img()

        # Bind mouse wheel for zoom (Windows/Linux)
        def on_mousewheel(event):
            """Handle mouse wheel events for zooming."""
            if event.delta > 0:
                zoom(event, zoom_in=True)
            else:
                zoom(event, zoom_in=False)
        canvas.bind("<MouseWheel>", on_mousewheel)
        # For MacOS
        canvas.bind("<Button-4>", lambda event: zoom(event, zoom_in=True))
        canvas.bind("<Button-5>", lambda event: zoom(event, zoom_in=False))

        root.bind("<Key>", lambda event: self.on_key(event, root))
        root.mainloop()

    def run(self):
        """Run the labeling process for all images."""
        while self.index < len(self.sorted_file_list):
            filename = self.sorted_file_list[self.index]
            self.current_key = None
            self.redo = False

            # Loop for redo
            while True:
                self.label_image(filename)
                if self.redo and self.leafdata:
                    self.leafdata.pop()
                    self.index -= 1
                    filename = self.sorted_file_list[self.index]
                    print(f"Redoing label for {filename}")
                    self.current_key = None
                    self.redo = False  # Reset redo flag for next label_image
                    continue  # Stay in redo loop
                break  # Exit redo loop if not redoing

            if self.current_key:
                self.leafdata.append(
                    (filename.rstrip(".png"), self.current_key))
                print(self.leafdata[-1])
                self.index += 1
            else:
                print("No key pressed, skipping or exiting.")
                break

        return self.leafdata


if __name__ == "__main__":
    check_alldownloaded()
    labeler = ImageLabeler(IMG_DIR, STARTFROM)
    try:
        ldata = labeler.run()
        ldata_df = pd.DataFrame(ldata, columns=["species", "shape"])
        print(ldata_df)
        last_n = ldata_df["species"].iloc[-1]
        last_n = re.search(r"\d+", last_n).group() if last_n else "0"
        ldata_df.to_csv(f"{WD}img_labels_{STARTFROM}_{last_n}.csv",
                        index=False)
    except Exception:
        traceback.print_exc()

# if __name__ == "__main__":
#     species_full = pd.read_csv(WD + FILE)
#     check_alldownloaded(species_full)

#     ddgi_path = WD + "ddgi_sample_" + FILE.split(".")[0]
#     print(ddgi_path)

#     if os.path.isdir(WD + "ddgi_sample_1"):
#         shutil.rmtree(WD + "ddgi_sample_1")
#     os.mkdir(WD + "ddgi_sample_1")

#     sorted_file_list = sorted(
#         os.listdir(WD + "Naturalis_multimedia_eud_sample_10-09-24_zuntini_intercept"),
#         # os.listdir("download_imgs"),
#         key=lambda x: int(re.search(r"\d+", x).group()),
#     )
#     for filename in sorted_file_list[STARTFROM:]:
#         if filename.endswith(".png"):
#             imgpath = WD + os.path.join(
#                 "Naturalis_multimedia_eud_sample_10-09-24_zuntini_intercept", filename
#             )
#             # imgpath = os.path.join("download_imgs", filename)
#             query = re.sub(r"\d+", "", filename.rstrip(".png"))
#             print(filename)
#             download_duckduckgo_images(query)
