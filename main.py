from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np

model = None
class_names = [
    'bacterial spot', 'early blight', 'healthy', 'late blight', 'leaf mold', 'mosaic virus', 'septoria leaf spot', 'spider mites', 'target spot', 'yellow leaf curl'
    ]

class_description = [
    # 1. bacterial_spot
    'Penyakit bacterial spot terjadi akibat infeksi oleh empat spesies bakteri dari genus Xanthomonas. Manifestasi awal penyakit ini adalah munculnya lesi berukuran kecil yang berwarna kuning pada daun muda, yang selanjutnya mengalami perkembangan menjadi lesi basah dan memiliki sifat berminyak pada daun yang sudah lebih tua, ditandai dengan perubahan warna menjadi cokelat hingga merah kecoklatan',
    # 2. early_blight
    'Early Blight adalah penyakit yang dapat menular yang memengaruhi tanaman tomat. Penyakit ini disebabkan oleh patogen jamur Alternaria linariae (syn. A. tomatophila) dan ditandai oleh adanya lesi pada batang, buah, dan daun tanaman tomat',
    # 3. healthy
    'Tanaman tomat yang dalam keadaan sehat atau Healthy dicirikan oleh daun yang memiliki tekstur lembut, berwarna hijau dengan intensitas warna yang berada dalam rentang dari hijau sedang hingga gelap, serta memiliki batang yang kuat dan kokoh',
    # 4. late_blight
    'Penyakit late blight, juga dikenal sebagai penyakit busuk daun, merupakan salah satu penyakit yang paling merusak pada tanaman tomat yang disebabkan oleh agen patogen bernama Phytophthora infestans (Mont.)',
    # 5. leaf_mold
    'Penyakit jamur daun tomat, yang juga dikenal sebagai leaf mold, merupakan penyakit yang sering ditemukan pada tanaman tomat. Kejadian penyakit ini umumnya terkait dengan lingkungan yang memiliki suhu tinggi dan kelembaban tinggi, kondisi yang mendukung penyebaran penyakit dan pertumbuhan patogen dengan cepat. Penyakit ini disebabkan oleh patogen Cladosporium fulvum (C. fulvum)',
    # 6. mosaic_virus
    'Mosaic Virus merupakan jenis virus tumbuhan yang memiliki kemampuan penyebaran yang sangat tinggi. Virus ini memiliki potensi untuk menginfeksi tanaman tomat, baik yang tumbuh di dalam rumah kaca maupun di lingkungan terbuka. Efek dari infeksi Mosaic virus termasuk terjadinya perubahan warna daun menjadi belang hijau pada daun yang masih muda maupun daun yang sudah tua, pertumbuhan tanaman yang terhambat, dan terkadang distorsi pada bentuk daun',
    # 7. septoria_leaf_spot
    'Penyakit Septoria Leaf Spot disebabkan oleh infeksi jamur Septoria lycopersici. Gejala khasnya adalah timbulnya lesi awal berbentuk bintik-bintik kecil yang di keadaan lembab, yang pada tahap selanjutnya berkembang menjadi bintik-bintik yang berbentuk lingkaran dengan diameter sekitar 1/8 inci',
    # 8. spider_mites
    'Spider Mites merupakan patologi yang diinduksi oleh serangga arachnid invasif, yang dicirikan oleh perubahan warna daun menjadi krem hingga kekuningan dengan munculnya bercak-bercak. Spider Mites dapat terdeteksi pada kedua sisi daun',
    # 9. target_spot
    'Penyakit target spot adalah kondisi yang dimulai dengan perkembangan lesi kecil berwarna gelap yang dilanjutkan denagn membesar membentuk lesi berwarna coklat muda dengan pola berpusat sama', 
    # 10. yellow_leaf_curl
    'Ciri khas pada tanaman yang mengalami infeksi penyakit Yellow leaf curl adalah adanya gejala seperti stunting (pertumbuhan terhambat), klorosis daun yang melengkung ke arah atas, pengurangan ukuran daun, dan penurunan hasil produksi tomat. Penyakit ini disebarkan dari satu tanaman ke tanaman lainnya melalui serangga'
    ]
class_prevent =[
    # 1. bacterial_spot
    'Melakukan penanaman dengan bibit yang bebas penyakit merupakan hal yang penting dalam upaya pengendalian penyakit bakteri, sebab bakteri dapat dengan mudah berpindah ke bibit tanaman melalui benih yang terkontaminasi. Disarankan untuk mengurangi aktivitas penanganan tanaman, seperti pemangkasan dan pengikatan, pada tingkat yang minimal karena luka yang terbentuk akibat penanganan tersebut memberikan akses bagi bakteri untuk masuk ke dalam sistem tanaman',
    # 2. early_blight
    'Untuk pencegahan penyakit ini, dianjurkan untuk menggunakan benih yang telah diuji bebas patogen atau mengumpulkan benih hanya dari tanaman yang terbebas dari penyakit tersebut. Penting juga untuk mengendalikan pertumbuhan gulma yang menjadi inang atau vektor penyakit. Selain itu, penting untuk menerapkan praktik pemupukan yang sesuai dengan rekomendasi, menghindari pemupukan berlebihan dengan unsur kalium, dan memastikan tingkat nitrogen dan fosfor dalam tanah mencukupi',
    # 3. healthy
    'Tanaman tomat yang sehat atau Healthy membutuhkan makronutrien seperti nitrogen, fosfor, dan kalium, dan mikronutrien seperti magnesium, kalsium, dan seng',
    # 4. late_blight
    'Pengendalian penyakit busuk daun dalam konteks budidaya tomat melibatkan sejumlah praktik pertanian dan tindakan pengendalian. Hal ini mencakup implementasi rotasi tanaman dan masa bera (periode tanah tidak ditanami), eliminasi tanaman tomat yang telah terinfeksi penyakit busuk daun, dan penghindaran produk tomat yang telah terkontaminasi. Selain itu, aplikasi fungisida juga merupakan metode umum untuk mengurangi perkembangan penyakit ini',
    # 5. leaf_mold
    'Langkah - langkah yang dapat diterapkan dalam budidaya tanaman tomat untuk mengatasi penyakit leaf mold melibatkan serangkaian tindakan. Pertama, daun yang telah terinfeksi harus diidentifikasi dan dipotong, lalu daun-daun tersebut harus dibuang dengan jarak yang cukup jauh dari area pertanian untuk menghindari penularan penyakit. Selanjutnya, dapat digunakan fungisida organik, seperti ekstrak bawang putih, yang memiliki kemampuan dalam mengendalikan populasi jamur. Pemberian pupuk harus dilakukan sesuai dengan dosis yang telah ditetapkan dalam pedoman budidaya tanaman tomat, untuk memastikan pertumbuhan tanaman yang sehat dan kuat dalam menghadapi tekanan penyakit',
    # 6. mosaic_virus
    'Untuk mencegah Mosaic Virus, disarankan untuk mengadopsi beberapa tindakan pencegahan yang disarankan. Pertama, pemilihan bibit yang berasal dari tanaman yang sehat dan tidak pernah terjangkit virus adalah langkah penting. Jika tanaman sudah terinfeksi mosaic virus, tindakan isolasi dapat diterapkan dengan mengelilinginya menggunakan kantong plastik yang diisi dengan pupuk kompos. Selain itu, tindakan isolasi juga dapat dilakukan dengan menjaga jarak fisik yang cukup jauh antara tanaman tomat yang terinfeksi dan tanaman tomat lainnya untuk meminimalkan risiko penularan virus',
    # 7. septoria_leaf_spot
    'Pencegahan penyakit Septoria Leaf Spot dapat dilakukan melalui tindakan-tindakan berikut. Pertama, mempertahankan jarak yang memadai antara tanaman tomat untuk memfasilitasi pengeringan daun yang lebih cepat. Selanjutnya, penyiraman pangkal tanaman tomat sebaiknya dilakukan pada waktu pagi guna mengurangi kelembaban dan potensi daun yang basah. Penting untuk menghindari perawatan tanaman tomat ketika daunnya dalam kondisi basah, karena tindakan ini dapat meminimalkan risiko penyebaran mikroorganisme penyebab penyakit',
    # 8. spider_mites
    'Untuk mencegah dan mengatasi hama tungau laba- laba atau Spider Mites pada tanaman, perlu menghilangkan tanaman inang umum seperti nightshade liar dan bayam, serta menjaga tanaman agar tidak mengalami kekurangan air yang dapat mendorong wabah hama tersebut. Penggunaan pestisida berbahan dasar minyak, seperti minyak nabati, minyak hortikultura, atau sabun murni, dapat membantu, dengan syarat petunjuk penggunaan yang sesuai dengan label produk diikuti. Pengaplikasian pestisida sebaiknya pada bagian bawah daun untuk mencapai tungau',
    # 9. target_spot
    'Untuk pencegahan dan penanganan penyakit Target Spot, disarankan melakukan pembentukan struktur tanaman dengan memangkas beberapa cabang dari bagian bawah tanaman untuk meningkatkan sirkulasi udara di pangkalan. Selain itu, daun bagian bawah yang terinfeksi sebaiknya segera dipotong dan dibakar ketika gejala penyakit muncul, terutama setelah panen batang buah bagian bawah. Kondisi lingkungan yang lembap dan hangat mendukung perkembangan penyakit ini, sehingga penggunaan fungisida diperlukan untuk mencapai kontrol yang efektif', 
    # 10. yellow_leaf_curl
    'Penyakit Yellow Leaf Curl dapat ditekan melalui metode pengendalian kimia. Setelah terjangkit oleh virus, tidak ada pengobatan yang efektif untuk infeksi tersebut, oleh karena itu, untuk mengelola populasi serangga yang berpotensi menyebarkan virus, digunakan insektisida dari keluarga piretroid. Insektisida ini dapat diterapkan sebagai pembasmi tanah atau semprotan selama tahap pembibitan tanaman untuk mengurangi populasi serangga yang berpotensi menyebabkan penyebaran virus'
    ]

BUCKET_NAME = "models_tomatify_numpang"

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

def load_model():
    global model
    if model is None:
        download_blob(
            BUCKET_NAME,
            "models/model.h5",
            "/tmp/model.h5",
        )
        model = tf.keras.models.load_model("/tmp/model.h5")

def predict(request):
    load_model()

    image = request.files["file"]

    image = np.array(
        Image.open(image).convert("RGB").resize((256, 256)) # image resizing
    )

    image = image / 255.0 # normalize the image in the range of 0 to 1

    img_array = tf.expand_dims(image, 0)
    predictions = model.predict(img_array)

    print("Predictions:", predictions)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    disease_description = class_description[np.argmax(predictions[0])]
    disease_prevention = class_prevent[np.argmax(predictions[0])]

    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "POST"
    }

    return {"kelas": predicted_class, "confidence": confidence, "description": disease_description, "prevention": disease_prevention}, 200, headers
