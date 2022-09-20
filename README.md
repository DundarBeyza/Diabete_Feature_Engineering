# Diabete_feature_engineering
Applying feature engineering to diabetes dataset
<h1 align="center">Hi 👋, I'm Beyza Dundar</h1>
<p align="left"> <img src="https://komarev.com/ghpvc/?username=dundarbeyza&label=Profile%20views&color=0e75b6&style=flat" alt="dundarbeyza" /> </p>

<h3 align="left">Connect with me:</h3>
<p align="left">
<a href="https://linkedin.com/in/beyza-dundar" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/linked-in-alt.svg" alt="beyza-dundar" height="30" width="40" /></a>
<a href="https://kaggle.com/beyzadundar" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/kaggle.svg" alt="beyzadundar" height="30" width="40" /></a>
<a href="https://medium.com/@beyzadndar" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/medium.svg" alt="@beyzadndar" height="30" width="40" /></a>
</p>

<h3 align="left">Languages and Tools:</h3>
<p align="left"> <a href="https://azure.microsoft.com/en-in/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/microsoft_azure/microsoft_azure-icon.svg" alt="azure" width="40" height="40"/> </a> <a href="https://www.docker.com/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/docker/docker-original-wordmark.svg" alt="docker" width="40" height="40"/> </a> <a href="https://hadoop.apache.org/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/apache_hadoop/apache_hadoop-icon.svg" alt="hadoop" width="40" height="40"/> </a> <a href="https://www.adobe.com/in/products/illustrator.html" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/adobe_illustrator/adobe_illustrator-icon.svg" alt="illustrator" width="40" height="40"/> </a> <a href="https://www.microsoft.com/en-us/sql-server" target="_blank" rel="noreferrer"> <img src="https://www.svgrepo.com/show/303229/microsoft-sql-server-logo.svg" alt="mssql" width="40" height="40"/> </a> <a href="https://www.oracle.com/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/oracle/oracle-original.svg" alt="oracle" width="40" height="40"/> </a> <a href="https://pandas.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/2ae2a900d2f041da66e950e4d48052658d850630/icons/pandas/pandas-original.svg" alt="pandas" width="40" height="40"/> </a> <a href="https://www.photoshop.com/en" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/photoshop/photoshop-line.svg" alt="photoshop" width="40" height="40"/> </a> <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a> <a href="https://scikit-learn.org/" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="scikit_learn" width="40" height="40"/> </a> <a href="https://seaborn.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://seaborn.pydata.org/_images/logo-mark-lightbg.svg" alt="seaborn" width="40" height="40"/> </a> </p>

Veri Setini Tanımak 

#Problem:Özellikleri belirtildiğinde kişilerin diyabet hastası olup olmadıklarını tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi
istenmektedir.Modeli geliştirmeden önce gerekli olan veri analizi ve özellik mühendisliği adımlarını gerçekleştirmemiz beklenmektedir.

#Veriseti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin parçasıdır.
#ABD'deki Arizona Eyaleti'nin en büyük 5.şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan Pima Indian kadınları
#üzerinde yapılan diyabet araştırması için kullanılan verilerdir.768 gözlem ve 8 sayısal bağımsız değişkenden oluşmaktadır.
#Hedef değişken "outcome" olarak belirtilmiş olup;1 diyabet test sonucunun pozitif oluşunu,0 ise negatif oluşunu belirtmektedir.

#Değişkenler

#Pregnancies: Hamilelik sayısı
#Glucose:Glikoz
#BloodPressure:Kan basıncı(Diastolic(KüçükTansiyon))
#SkinThickness:Cilt Kalınlığı
#Insulin:İnsülin.
#BMI:Beden kitle indeksi.
#DiabetesPedigreeFunction:Soyumuzdaki kişilere göre diyabet olma ihtimalimizi hesaplayan bir fonksiyon.
#Age:Yaş(yıl)
#Outcome:Kişinin diyabet olup olmadığı bilgisi.Hastalığa sahip(1) yada değil(0).


#GÖREV1:KEŞİFCİ VERİ ANALİZİ

#Adım1:Genel resmi inceleyiniz.
#Adım2:Numerik ve kategorik değişkenleri yakalayınız.
#Adım3:Numerik ve kategorik değişkenlerin analizini yapınız.
#Adım4:Hedef değişken analizi yapınız.
#Adım5:Aykırı gözlem analizi yapınız.
#Adım6:Eksik gözlem analizi yapınız.
#Adım7:Korelasyon analizi yapınız.

#GÖREV2:FEATURE ENGINEERING

#Adım1:Eksik ve aykırı değerler için gerekli işlemleri yapınız.
Verisetinde eksik gözlem bulunmamakta ama Glikoz,Insulin vb. değişkenlerde 0 değeri içeren gözlem birimleri eksikdeğeri ifade ediyor olabilir.Örneğin;bir kişinin glikoz veya insulin değeri 0 olamayacaktır.Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp sonrasında eksik değerlere işlemleri uygulayabilirsiniz.
#Adım2:Yeni değişkenler oluşturunuz.
#Adım3:Encoding işlemlerini gerçekleştiriniz.
#Adım4:Numerik değişkenler için standartlaştırma yapınız.
#Adım5:Model oluşturunuz.

