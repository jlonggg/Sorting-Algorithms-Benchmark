# Viết hàm QuickSort
def partition(arr, low, high):
    piv_idx = (low + high) // 2
    pivot = arr[piv_idx]
    i = low - 1
    j = high + 1 

    while True:
        i += 1
        while arr[i] < pivot:
            i += 1
        
        j -= 1
        while arr[j] > pivot:
            j -= 1
        
        if i >= j:
            return j
        
        arr[i], arr[j] = arr[j], arr[i]

def QuickSort(arr, low, high):
    if low < high:
        piv = partition(arr, low, high)
        QuickSort(arr, low, piv)
        QuickSort(arr, piv + 1, high)
    
# Viết hàm MergeSort
def merge_sort(arr, start_index, end_index):
    if start_index >= end_index:
        return

    mid_index = start_index + (end_index - start_index) // 2

    merge_sort(arr, start_index, mid_index)
    
    merge_sort(arr, mid_index + 1, end_index)

    merge(arr, start_index, mid_index, end_index)


def merge(arr, start, mid, end):
    left_half = arr[start : mid + 1]
    right_half = arr[mid + 1 : end + 1]

    i = 0 
    j = 0 
    k = start 

    while i < len(left_half) and j < len(right_half):
        if left_half[i] <= right_half[j]:
            arr[k] = left_half[i]
            i += 1
        else:
            arr[k] = right_half[j]
            j += 1
        k += 1

    while i < len(left_half):
        arr[k] = left_half[i]
        i += 1
        k += 1

    while j < len(right_half):
        arr[k] = right_half[j]
        j += 1
        k += 1

# Viết hàm HeapSort
def heapify(arr, n, i):
    current = i

    while True:
        largest = current
        left = 2 * current + 1
        right = 2 * current + 2

        if left < n and arr[left] > arr[largest]:
            largest = left

        if right < n and arr[right] > arr[largest]:
            largest = right

        if largest == current:
            break
        
        arr[current], arr[largest] = arr[largest], arr[current]
        
        current = largest

def HeapSort(arr):
    n = len(arr)

    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i] 
        
        heapify(arr, i, 0)

import time
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Tăng giới hạn đệ quy
sys.setrecursionlimit(10**7)

if __name__ == "__main__":
    # ==========================================
    # 1. TẠO DỮ LIỆU (DATA GENERATION)
    # ==========================================
    N = 1_000_000
    dataset = []
    labels = []

    print(f"--- Đang khởi tạo bộ dữ liệu ({N} phần tử/dãy) ---")

    for i in range(10):
        if i < 5:
            arr = np.random.uniform(-100, 100, N)
            dtype_label = "Float"
        else:
            arr = np.random.randint(-100, 100, N)
            dtype_label = "Int"

        if i == 0:
            arr = np.sort(arr)
            status_label = "Sorted (Asc)"
        elif i == 5:
            arr = np.sort(arr)[::-1]
            status_label = "Sorted (Desc)"
        else:
            status_label = "Random"

        dataset.append(arr)
        labels.append(f"{dtype_label} - {status_label} ({i})")

    # Kiểm tra kết quả
    print(f"Đã tạo {len(dataset)} dãy.")
    print(f"Dãy 1 (Số thực tăng dần): {dataset[0][:5]} ...")
    print(f"Dãy 6 (Số nguyên giảm dần): {dataset[5][:5]} ...")
    print("Đã tạo xong 10 dãy dữ liệu.\n")
    
    # ==========================================
    # 2. WRAPPER FUNCTIONS
    # ==========================================
    def run_quicksort(arr): QuickSort(arr, 0, len(arr) - 1)
    def run_mergesort(arr): merge_sort(arr, 0, len(arr) - 1)
    def run_heapsort(arr): HeapSort(arr)
    def run_python_sort(arr): list(arr).sort() 
    def run_numpy_sort(arr): np.sort(arr, kind='quicksort')

    algorithms = {
        "QuickSort": run_quicksort,
        "HeapSort": run_heapsort,
        "MergeSort": run_mergesort,
        "sort (C++)": run_python_sort,
        "sort (numpy)": run_numpy_sort
    }
    # ==========================================
    # 3. CHẠY THỬ NGHIỆM
    # ==========================================
    results = []
    header = f"{'Dữ liệu':<10} | {'QuickSort':<12} | {'HeapSort':<12} | {'MergeSort':<12} | {'sort (C++)':<12} | {'sort (numpy)':<12}"
    print(header)
    print("-" * len(header))

    for i, data_original in enumerate(dataset):
        row_result = {"Dữ liệu": i + 1}
        print(f"{i + 1:<10} | ", end="", flush=True)

        for algo_name, algo_func in algorithms.items():
            arr_copy = data_original.copy()
            start_time = time.time()
            algo_func(arr_copy)
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            row_result[algo_name] = round(duration_ms, 2)
            print(f"{duration_ms:9.2f} ms | ", end="", flush=True)
        
        print("")
        results.append(row_result)
    # ==========================================
    # 4. BÁO CÁO & LƯU FILE
    # ==========================================
    df = pd.DataFrame(results)
    
    # Tính trung bình
    mean_values = df.iloc[:, 1:].mean()
    mean_row = mean_values.to_dict()
    mean_row["Dữ liệu"] = "Trung bình"
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    df.set_index("Dữ liệu", inplace=True)
    
    print("\n" + "="*30)
    print("BẢNG KẾT QUẢ (ms)")
    print("="*30)
    print(df)

    # === LƯU FILE CSV ===
    csv_filename = "ket_qua_sap_xep.csv"
    df.to_csv(csv_filename)
    print(f"\n[OK] Đã lưu file CSV tại: {os.path.abspath(csv_filename)}")

    # === VẼ & LƯU BIỂU ĐỒ ===
    df_chart = df.drop("Trung bình")
    ax = df_chart.plot(kind='bar', figsize=(14, 8), width=0.85)
    
    plt.title("Kết quả thử nghiệm (1 triệu phần tử)", fontsize=16)
    plt.ylabel("Thời gian (ms)", fontsize=12)
    plt.xlabel("Dữ liệu (1-10)", fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title="Thuật toán")

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.0f}', (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='bottom', xytext=(0, 5), 
                        textcoords='offset points', fontsize=7, rotation=90)

    plt.tight_layout()
    
    img_filename = "bieu_do_so_sanh.png"
    plt.savefig(img_filename)
    print(f"[OK] Đã lưu ảnh biểu đồ tại: {os.path.abspath(img_filename)}")
    
    plt.show()