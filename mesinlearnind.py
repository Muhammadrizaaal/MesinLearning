import numpy as np

X_train = np.array([[40, 5, 60],
                    [50, 8, 40],
                    [50, 7, 30],
                    [70, 4, 60],
                    [80, 4, 80],
                    [60, 6, 60]])
y_train = np.array(['Jelek', 'Bagus', 'Jelek', 'Bagus', 'Bagus', 'Bagus'])

X_test = np.array([50, 3, 40])


distances = np.sqrt(np.sum((X_train - X_test)**2, axis=1))

#untuk k=3,jika kita ingin mengubah k menjadi 4 dan 5 kita ubah saja bagian k di bawah ini
k = 3 
nearest_indices = np.argsort(distances)[:k]  
nearest_neighbors = y_train[nearest_indices] 


unique_classes, counts = np.unique(nearest_neighbors, return_counts=True)
prediction = unique_classes[np.argmax(counts)]


print(f"Kelas prediksi untuk data uji adalah '{prediction}'.")
