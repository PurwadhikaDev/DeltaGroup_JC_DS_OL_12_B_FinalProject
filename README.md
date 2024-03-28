# Hotel Booking Demand Prediction (Classification Model)
Data Source:
Nuno Antonio, Ana Almeida, and Luis Nunes for the publication Data in Brief, Volume 22, February 2019, and it can be downloaded in [Sumber](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand/code)

Tableau Dashboard [link](https://public.tableau.com/app/profile/anna.charieninna7072/viz/HotelBookingDemand_17114718998920/Dashboard1) 

<br>Jika ada image yang tidak bisa diload, silahkan kunjungi [link ini](https://nbviewer.org/github/PurwadhikaDev/DeltaGroup_JC_DS_OL_12_B_FinalProject/blob/main/Hotel%20Booking%20Demand.ipynb)

# **Hotel Booking Demand***

Context :
Beberapa tahun lalu, industri hotel telah mengalami perubahan dengan sebagian besar pemesanan kini dilakukan secara online seperti Booking.com (sumber). Akibatnya, pelanggan telah terbiasa dengan kebijakan pembatalan gratis. Bahkan, sebuah studi yang dilakukan oleh D-Edge Hospitality Solutions menemukan bahwa tingkat pembatalan melalui semua saluran telah meningkat sebesar 6% selama empat tahun terakhir, mencapai hampir 40% pada tahun 2018 (sumber) dan memperoleh kerugian tertinggi pada tahun 2016 yaitu mencapai 2.8 juta Euro.
Menganalisis kinerja bisnis adalah komponen penting untuk mencapai kesuksesan bagi perusahaan. Melalui analisis yang cermat, perusahaan dapat mengevaluasi kinerja bisnis mereka.
Dalam konteks industri perhotelan, memahami perilaku pelanggan menjadi sangat penting. Pemahaman ini memungkinkan perusahaan untuk menemukan faktor-faktor yang memengaruhi pelanggan saat mereka memesan hotel. Selain itu, ini memungkinkan perusahaan untuk mengenali produk atau layanan mana yang kurang berhasil di pasar sehingga layanan bisa lebih customer centric.
Analisis yang cermat selanjutnya dapat berguna untuk merancang strategi bisnis yang efektif, yang pada akhirnya meningkatkan customer experience dan membuka jalan menuju bisnis yang berkelanjutan.
Target :
0 : Tidak melakukan pembatalan pemesanan
1 : Melakukan pembatalan pemesanan

Problem Statement :
Pemahaman yang mendalam tentang perilaku pembatalan pelanggan juga dapat membantu hotel meningkatkan pengelolaan kapasitas dan inventarisasi kamar, mengoptimalkan pemesanan, dan meningkatkan kinerja operasional secara keseluruhan. Dengan demikian, fokus pada analisis perilaku pembatalan dapat membantu hotel meningkatkan pengalaman pelanggan, memperkuat reputasi merek, dan mencapai keunggulan bersaing dalam industri perhotelan.
Adapun kita mempunyai beberapa business question:
1. Bagaimana kebiasaan pelanggan yang melakukan booking dan berapa total kerugian yang diperoleh akibat pembatalan pemesanan?
2. Apa saja faktor yang mempengaruhi pelanggan yang cenderung membatalkan pemesanan, dan apa langkah-langkah strategis yang dapat diambil untuk mengurangi jumlah pembatalan?
3. Bagaimana kita dapat meningkatkan efektivitas kebijakan untuk meningkatkan jumlah booking sehingga menjadi pendapatan dan meningkatkan pelayanan untuk kepuasan pelanggan?
4. Bagaimana kita dapat meminimalisir peluang calon customer untuk canceled dengan model prediktif yang dibangun sehingga meningkatkan awareness kepada stakeholders sehingga dapat mengubah potensi canceled calon customer menjadi potensi booking calon customer menjadi revenue?

Analytic Approach :
Pendekatan analitik yang akan kami gunakan adalah pendekatan prediktif menggunakan teknik machine learning, khususnya dalam klasifikasi. Kami akan menggunakan dataset yang ada untuk melatih model yang dapat memprediksi apakah suatu pemesanan akan dibatalkan atau tidak berdasarkan atribut-atribut yang relevan. Langkah-langkah utama dalam pendekatan ini mencakup pemrosesan data untuk persiapan, pemilihan fitur, pemodelan dengan algoritma yang sesuai dengan benchmarking model, seperti Random Forest, Logistic Regression, XGBoost, LightGBM dan lain sebagainya validasi model menggunakan metrik evaluasi yang tepat, dan penyesuaian model jika diperlukan. Tujuan utama dari pendekatan ini adalah untuk menghasilkan model yang dapat memberikan prediksi akurat untuk membantu hotel dalam pengelolaan reservasi dan pengambilan keputusan yang lebih baik.

Metric Evaluation :
Dalam konteks prediksi pembatalan pelanggan hotel, kita ingin memastikan model dapat mengidentifikasi pelanggan yang cenderung membatalkan reservasi mereka dengan akurasi yang tinggi tanpa memicu terlalu banyak prediksi yang salah tentang pembatalan. Oleh karena itu, ROC AUC score adalah metrik evaluasi yang cocok untuk memberikan pemahaman yang lengkap tentang kinerja model dalam kasus ini. Berikut referensi penggunaan metrik evaluasi ROC AUC pada studi kasus yang sama.[Sumber](https://www.kaggle.com/code/jcaliz/ps-s03e07-a-complete-eda#Submission)

RUC-ROC adalah area di bawah kurva ROC (Receiver Operating Characteristic). Kurva ROC menggambarkan tingkat True Positive Rate (sensitivitas) terhadap False Positive Rate (1 — Specificity) pada berbagai threshold yang digunakan model untuk memprediksi instance. Nilai AUC yang lebih tinggi menunjukkan bahwa model lebih baik dalam membedakan antara instance positif dan negatif. Rumus untuk luas dibawah kurva adalah integral dari fungsinya

- Rumus True Positive Rate (TPR):

$TPR = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$

- Rumus False Positive Rate (FPR):

$FPR = \frac{\text{False Positives}}{\text{False Positives} + \text{True Negatives}}$

<center>
  <img src="https://miro.medium.com/v2/resize:fit:936/format:webp/1*uGn46fPbYz65HAWLFiVqDg.jpeg" alt="Rumus ROC AUC">
</center>

Data Preprocessing & Feature Engineering :

- Feature Encoding : One Hot Encoding & Binary Encoding
- Feature Scaling : Robust Scaler
- Feature Selection : Feature Correlation
 
Best Model :
Setelah dengan benchmarking pada data training dan data testing dengan berbagai model didapatkan model XGBoost memiliki ROC AUC score paling baik sebesar ROC AUC Score Tuned XGBoost : 0.915389768204184.

XGBoost dikenal karena kinerja yang cepat, skalabilitas yang baik, dan kemampuan untuk menghasilkan model yang sangat baik dalam berbagai jenis data.
Prinsip kerja XGBoost adalah dengan membangun serangkaian model prediktif sederhana, biasanya pohon keputusan (decision trees), dan menggabungkan prediksi dari setiap model untuk meningkatkan akurasi prediksi secara keseluruhan. Proses ini dilakukan secara iteratif, dengan setiap iterasi (atau "boosting round"), model baru ditambahkan untuk memperbaiki kesalahan prediksi yang dibuat oleh model sebelumnya.
dengan best parameter untuk XGBoost adalah:
- subsample : 0.8
- random_state : 0
- n_estimators : 200
- max_depth : 17
- learning_rate : 0.08

Feature Importance :
Model yang dihasilkan sangat dipengaruhi oleh beberapa fitur 3 fitur paling berpengaruh dalam model memprediksi kemungkinan pelanggan canceled adalah
- Kebutuhan tempat parkir
- Jenis Deposit Non-Refundable
- Riwayat Cancel sebelumnya

Limitasi Model :
Model ini hanya berlaku pada rentang data yang digunakan pada pemodelan ini yaitu :
lead time yaitu range antara 0-737
arrival date week number range antara 1-53
adults range dari 0-55
children range dari 0-10
babies range dari 0-10
previous cancellations range dari 0-26
previous bookings not canceled range antara 0-72
booking_changes range antara 0-18
days_in_waiting_list range antara 0-391
adr range antara 0-226.84
required_car_parking_spaces range antara 0-8
total_of_special_requests range antara 0-5

# Conclusions

TANPA MENGGUNAKAN MACHINE LEARNING
Dengan asumsi tarif harian rata-rata hotel sebesar 100 Euro, maka simulasinya sebagai berikut:
100 Euro x 61.675 (Customer Booking) = 61.675.000 Euro (Revenue)
100 Euro X 23.039 (Customer Canceled) = 23.039.000 Euro (Lost Revenue)
DENGAN MENGGUNAKAN MACHINE LEARNING
Dengan asumsi yang sama jika menggunakan machine learning maka simulasinya adalah sebagai berikut:
(TP+TN): (11280+3159) x 100 Euro = 14.439.000 Euro
(FP+FN): (1055+1449) x 100 Euro = 2.504.000 Euro
Dikarenakan kita menggunakan test data sebesar 20% maka, perhitungan asumsi dikali 5 yaitu sebagai berikut :
14.439.000 Euro x 5 = 72.195.000 Euro (Revenue)
2.504.000 Euro x 5 = 12.520.000 Euro (Lost Revenue)
PERBANDINGAN PENGGUNAAN MACHINE LEARNING
Sebelum pakai ML: Hotel Revenue sebesar 61.675.000 Euro.
Setelah menggunakan ML: Hotel Revenue menjadi 72.195.000Euro.
Sehingga Profit yang dihasilkan dari selisih Revenue adalah sebesar:
72.195.000 Euro - 61.675.000 Euro = 10.520.000 Euro.
Dapat disimpulkan bahwa Machine Learning dengan menggunakan algoritma XGBoost setelah tuning berhasil menghasilkan profit 10.230.000 Euro dengan persentase sebesar 12.42%.

# Recommendations

For Model :
Mengumpulkan lebih banyak data khususnya pada minority class.
Menambahkan parameter lain dalam hyperparameter tuning.
Menambahkan ID customer atau booking ID untuk memastikan dan mengetahui data yang duplikat.
Meminimalisir kesalahan penulisan data dan memastikan data yang diperoleh tidak ada yang kosong atau tidak terisi.
Mencoba ML algorithm diluar ML algorithm yang dipakai di project ini, dan mencoba dengan hyperparamater tuning kembali seperti SMOTENC dan lain sebagainya.

For Business:
Mengembangkan strategi untuk meningkatkan ketersediaan lahan parkir dapat membantu mengurangi pembatalan, seperti menambah fasilitas parkir tambahan atau menawarkan paket parkir tambahan.
Menerapkan kebijakan deposit yang lebih ketat atau meningkatkan nilai deposit dapat membantu mengurangi kemungkinan pembatalan, karena pelanggan akan lebih berpikir dua kali sebelum membatalkan pemesanan mereka.
Memperhatikan riwayat pembatalan sebelumnya dapat membantu dalam menilai risiko pembatalan untuk setiap pemesanan.
Fokus pada upaya pemasaran yang lebih kuat untuk menarik lebih banyak pemesanan di hotel resor (karena pemesanannya masih cenderung lebih sedikit) serta mengembangkan kebijakan pembatalan yang lebih menarik untuk menarik minat pelanggan.
Menawarkan paket atau promosi khusus untuk menginsentifkan tamu untuk memperpanjang masa menginap mereka di hotel resort, seperti diskon untuk menginap lebih lama.
Meningkatkan kerjasama dengan agen perjalanan online atau mengembangkan strategi pemasaran dengan menggunakan gimmick-gimmick.
Mengidentifikasi tren perjalanan dari negara-negara tertentu dan menyesuaikan strategi pemasaran untuk menarik lebih banyak tamu dari negara-negara yang potensial.
Fokus pada tamu lokal, karena tamu dari kedua hotel didominasi oleh tamu lokal (Portugal) dan sekitar benua Eropa sehingga sangat penting untuk memberikan kenyamanan dan impresi menginap yang baik untuk mengurangi tingkat pembatalan.
Menyesuaikan strategi harga dan promosi untuk memaksimalkan pendapatan dari tipe kamar yang paling diminati (tipe A) agar kamar lainnya agar dapat bersaing dengan tipe kamar A.


