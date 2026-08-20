# IARONE · 3D model + bilek AR

Statik bir site. Build adımı yok — klasörü herhangi bir statik hosting'e (Vercel, Netlify,
GitHub Pages) olduğu gibi yükleyince çalışır.

> **Kamera erişimi `https://` ister.** Yerelde `http://localhost` da kabul edilir,
> `file://` çalışmaz.

```bash
python -m http.server 8080     # -> http://localhost:8080
```

## Sayfalar

| | |
|---|---|
| `index.html` | Ürün sayfası: döndürülebilir 3D görüntüleyici + iki AR butonu |
| `ar.html` | **Bilekte dene** — kamera + el takibi, uygulama kurulumu yok |

`ar.html` hata ayıklama parametreleri:

| parametre | ne yapar |
|---|---|
| `?demo=1` | Gerçek kamera yerine sentetik el; kamerasız test |
| `?arm=1` | Kol silindirini görünür yapar (hizalama kontrolü) |
| `?dbg=1` | Poz / derinlik bilgisi |
| `?hand=left`, `?palm=1`, `?pose=still\|tilt\|turn` | demo varyantları |

## Model

`iarone-watch-v1.glb` — 36.198 üçgen, 4,3 MB, 11 PBR materyali.
Gerçek dünya ölçeğinde (kasa Ø 40,0 mm), metre biriminde, Y-yukarı.
Kadran normali +Y, saat 12 yönü −Z, kurma kolu ve **kayış halkası ekseni +X**.

`iarone-watch.usdz` — iOS AR Quick Look için. Y-yukarı, metre, sıkıştırmasız ve
64 bayt hizalı (Apple şartı). Safir cam çıkarılmıştır: Quick Look ince şeffaf
yüzeyleri kötü çiziyor.

## Bilekte deneme nasıl çalışıyor

1. **MediaPipe HandLandmarker** eli bulur (21 nokta; 2B görüntü + metrik 3B).
2. `wrist.js` bilek çerçevesini kurar:
   - **El sırtı / avuç ayrımı 2B'de** yapılır: (bilek, işaret MCP, serçe MCP)
     üçgeninin dönüş yönü. Metrik noktaların z işareti güvenilmez olduğu için
     iki konvansiyon denenip bu ölçütle uyan seçilir.
   - **Derinlik**, elin metrik genişliği ÷ piksel genişliği oranından çıkar.
     Bu sayede kameranın gerçek görüş açısı bilinmese bile saat ekranda
     doğru büyüklükte görünür.
   - **Bilek çapası** ham WRIST noktası değildir — o, elin en titrek yeridir.
     Üç parmak boğumunun ortalamasından çıkılıp, boğum açıklığına oranı zamanla
     yumuşatılmış bir mesafe kullanılır.
3. **Kol, görünmez bir elips silindirle** temsil edilir; yalnızca derinlik yazar
   (`colorWrite: false`). Kayışın kolun arkasında kalan kısmı böylece gizlenir.
4. Poz **1-Euro filtresiyle** (Casiez et al., CHI 2012) yumuşatılır: durgunken
   kesme frekansı düşük (titremesiz), hareket ederken yüksek (gecikmesiz).

### Geometrinin can alıcı noktası

**Kol, kayış halkasının içinden geçer; saat 12–6 ekseni kola diktir.**

Kayış kapalı bir halkadır ve kol içinden geçer, dolayısıyla halkanın ekseni kolun
eksenidir. Kayış 12 ve 6 kulaklarına bağlıdır, yani bu iki nokta halkanın üzerindedir;
halka kola dik bir düzlemde olduğuna göre 12–6 ekseni de kola diktir. Saatçilikte
"lug-to-lug bileğimin **genişliğini** aşar mı" diye sorulması da bunu doğrular.

"Saat 12 ele bakar" sezgisi yanıltıcıdır — o, saati okurken bileği çevirdiğimiz için
oluşur. Bu yüzden modelde kol `+X` eksenidir.

## Kalibrasyon

`ar.html` içindeki **Ayar** düğmesi canlı kaydırıcılar açar: kola doğru, yükseklik,
bilek çevresi, döndürme (kol ekseni etrafında), saat boyutu; ayrıca **Tarafı çevir**
ve **Kurma kolu yönü**. Değerler tarayıcıda saklanır (`localStorage: iarone-fit`).

> `wrist.js` içindeki `FIT.outOfWrist` ile `tools/build.py` içindeki bilek elipsi
> birbirine bağlıdır. Kayış deriden yalnızca ~2,6 mm yukarıdadır; 2 mm'lik hata
> kayışı kolun içine gömer. İkisini birlikte güncelleyin.

## İlk yükleme boyutu

`ar.html` ilk açılışta ~23 MB indirir (11,5 MB wasm + 7,6 MB el modeli + 4,3 MB saat);
brotli ile tel üzerinde ~14–16 MB. Sonrası tarayıcı önbelleğinden gelir.
`vendor/tasks-vision/wasm/*_nosimd_*` yalnızca eski cihazlar içindir.

## tools/

Modeli üreten parametrik kaynak (Python + numpy/PIL/OpenCV):

```bash
python tools/build.py            # iarone_watch.glb
python tools/build.py --usdz     # USDZ kaynağı (cam olmadan)
blender -b -P tools/to_usdz.py -- <in.glb> <out_dir>
python tools/pack_usdz.py <out_dir> iarone-watch.usdz
```

`tools/logo_extract.py` + `logo_vector.py` logoyu referans üründen çıkarıp düz
çizgilere oturtur. **Referans ürün görseli bu repoya dahil edilmemiştir**; script'ler
onu `ref.png` adıyla bekler.

## Bilinen sınırlar

- Tek el takip edilir (`numHands: 1`).
- Kol için ortalama bir elips kullanılır; çok ince/kalın bileklerde "Ayar" gerekir.
- Sabit stüdyo aydınlatması; ortam ışığı tahmini yapılmaz.
- Gerçek cihazda kamerayla doğrulama kullanıcı tarafından yapılmaktadır.
