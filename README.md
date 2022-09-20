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

#Problem:Özellikleribelirtildiğindekişilerindiyabethastasıolupolmadıklarınıtahminedebilecekbirmakineöğrenmesimodeligeliştirilmesi
#istenmektedir.Modeligeliştirmedenöncegerekliolanverianaliziveözellikmühendisliğiadımlarınıgerçekleştirmenizbeklenmektedir.

#VerisetiABD'dekiUlusalDiyabet-Sindirim-BöbrekHastalıklarıEnstitüleri'ndetutulanbüyükverisetininparçasıdır.
#ABD'dekiArizonaEyaleti'ninenbüyük5.şehriolanPhoenixşehrindeyaşayan21yaşveüzerindeolanPimaIndiankadınları
#üzerindeyapılandiyabetaraştırmasıiçinkullanılanverilerdir.768gözlemve8sayısalbağımsızdeğişkendenoluşmaktadır.
#Hedefdeğişken"outcome"olarakbelirtilmişolup;1diyabettestsonucununpozitifoluşunu,0isenegatifoluşunubelirtmektedir.

#Pregnancies:Hamileliksayısı
#Glucose:Glikoz
#BloodPressure:Kanbasıncı(Diastolic(KüçükTansiyon))
#SkinThickness:CiltKalınlığı
#Insulin:İnsülin.
#BMI:Bedenkitleindeksi.
#DiabetesPedigreeFunction:Soyumuzdakikişileregörediyabetolmaihtimalimizihesaplayanbirfonksiyon.
#Age:Yaş(yıl)
#Outcome:Kişinindiyabetolupolmadığıbilgisi.Hastalığasahip(1)yadadeğil(0)


#GÖREV1:KEŞİFCİVERİANALİZİ
#Adım1:Genelresmiinceleyiniz.
#Adım2:Numerikvekategorikdeğişkenleriyakalayınız.
#Adım3:Numerikvekategorikdeğişkenlerinanaliziniyapınız.
#Adım4:Hedefdeğişkenanaliziyapınız.(Kategorikdeğişkenleregörehedefdeğişkeninortalaması,hedefdeğişkenegörenumerikdeğişkenlerinortalaması)
#Adım5:Aykırıgözlemanaliziyapınız.
#Adım6:Eksikgözlemanaliziyapınız.
#Adım7:Korelasyonanaliziyapınız.

#GÖREV2:FEATUREENGINEERING
#Adım1:Eksikveaykırıdeğerleriçingerekliişlemleriyapınız.VerisetindeeksikgözlembulunmamaktaamaGlikoz,Insulinvb.
#değişkenlerde0değeriiçerengözlembirimlerieksikdeğeriifadeediyorolabilir.Örneğin;birkişininglikozveyainsulindeğeri
#0olamayacaktır.BudurumudikkatealaraksıfırdeğerleriniilgilideğerlerdeNaNolarakatamayapıpsonrasındaeksikdeğerlere
#işlemleriuygulayabilirsiniz.
#Adım2:Yenideğişkenleroluşturunuz.
#Adım3:Encodingişlemlerinigerçekleştiriniz.
#Adım4:Numerikdeğişkenleriçinstandartlaştırmayapınız.
#Adım5:Modeloluşturunuz.
![image](https://user-images.githubusercontent.com/111129459/191240205-a28aa7d0-a2a4-427b-bc11-9baaaebc9bdb.png)
