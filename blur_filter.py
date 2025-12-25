# This Image Blur filter is developed by Omama Sajid, Saleha Arshad 
from mpi4py import MPI
from PIL import Image, ImageFilter
import numpy as np
from time import time
import matplotlib.pyplot as plt
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    img_file = input("Enter image filename (with extension): ")
    while True:
        blur_type = input("Enter blur type (gaussian/median/box): ").lower()
        if blur_type in ["gaussian", "median", "box"]:
            break
        else:
            print("Invalid blur type! Please enter gaussian, median, or box.")
    while True:
        try:
            radius = int(input("Enter blur radius (positive integer): "))
            if radius > 0:
                break
            else:
                print("Radius must be positive.")
        except:
            print("Enter a valid integer.")

    img = Image.open(img_file).convert("RGB")
    img_array = np.array(img)
    height, width, channels = img_array.shape
else:
    img_array = None
    height = None
    width = None
    channels = None
    blur_type = None
    radius = None

height = comm.bcast(height, root=0)
width = comm.bcast(width, root=0)
channels = comm.bcast(channels, root=0)
blur_type = comm.bcast(blur_type, root=0)
radius = comm.bcast(radius, root=0)

chunk_size = height // size
overlap = radius * 2  

if rank == 0:
    chunks = []
    for i in range(size):
        start = max(i * chunk_size - overlap, 0)
        end = min((i + 1) * chunk_size + overlap, height)
        chunks.append(img_array[start:end])
else:
    chunks = None

chunk = comm.scatter(chunks, root=0)

def apply_blur(image_array, blur_type, radius):
    chunk_img = Image.fromarray(image_array)
    if blur_type == "gaussian":
        blurred_chunk_img = chunk_img.filter(ImageFilter.GaussianBlur(radius=radius))
    elif blur_type == "median":
       
        size = max(3, radius * 2 + 1)  
        blurred_chunk_img = chunk_img.filter(ImageFilter.MedianFilter(size=size))
    elif blur_type == "box":
        blurred_chunk_img = chunk_img.filter(ImageFilter.BoxBlur(radius))
    else:
        blurred_chunk_img = chunk_img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(blurred_chunk_img)

comm.Barrier()
mpi_start = time()
blurred_chunk = apply_blur(chunk, blur_type, radius)
comm.Barrier()
mpi_end = time()
mpi_total_time = mpi_end - mpi_start

comm.Barrier()
compute_start = time()
blurred_chunk_compute = apply_blur(chunk, blur_type, radius)
comm.Barrier()
compute_end = time()
mpi_compute_time = compute_end - compute_start
if rank == 0:
    blurred_chunk_compute = blurred_chunk_compute[:chunk_size]
elif rank == size - 1:
    blurred_chunk_compute = blurred_chunk_compute[overlap:]
else:
    blurred_chunk_compute = blurred_chunk_compute[overlap:-overlap]
os.makedirs("chunks", exist_ok=True)
Image.fromarray(blurred_chunk_compute).save(f"chunks/chunk_rank{rank}.png")

gathered = comm.gather(blurred_chunk_compute, root=0)

if rank == 0:
    
    result_array = np.vstack(gathered)
    result_img = Image.fromarray(result_array)
    result_img.save("mpi_blurred_image.png")
    

    print(f"\nMPI processing time (total, includes communication): {mpi_total_time:.6f} s")
    print(f"MPI computation-only time: {mpi_compute_time:.6f} s")
    serial_start = time()
    result_serial = apply_blur(img_array, blur_type, radius)
    serial_end = time()
    serial_time = serial_end - serial_start
    Image.fromarray(result_serial).save("serial_blurred_image.png")

    print(f"Serial processing time: {serial_time:.6f} s")
    print("Serial blurred image saved as serial_blurred_image.png")
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_array)
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(result_array)
    plt.title("MPI Blurred Image")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("original_vs_blurred.png")
    plt.show()
    print("Original vs MPI blurred image saved as original_vs_blurred.png")
    labels = ["Serial", f"MPI Compute ({size} proc)", f"MPI Total ({size} proc)"]
    times = [serial_time, mpi_compute_time, mpi_total_time]

    plt.figure()
    plt.bar(labels, times)
    plt.title("Serial vs MPI Processing Time")
    plt.xlabel("Execution Type")
    plt.ylabel("Time (seconds)")
    plt.savefig("timing_graph.png")
    plt.show()
    print("Timing graph saved as timing_graph.png")
