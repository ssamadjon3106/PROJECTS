from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


LEGAL_DOCS_DIR = Path("legal_docs")


@dataclass(frozen=True)
class KnowledgeDocument:
    filename: str
    title: str
    content: str

    @property
    def path(self) -> Path:
        return LEGAL_DOCS_DIR / self.filename


CONSTITUTION_TEXT = """O‘ZBEKISTON RESPUBLIKASI KONSTITUTSIYASI ASOSIDA TUZILGAN SINTETIK MATN

1-bob. Inson huquqlari, sha’n va daxlsizlik

1-modda. Qadr-qimmat va shaxsiy erkinlik
Har bir shaxsning sha’ni, qadr-qimmati, shaxsiy erkinligi va daxlsizligi davlat tomonidan e’tirof etiladi hamda himoya qilinadi. Hech kim qonunda nazarda tutilmagan asoslar bilan erkinlikdan mahrum etilishi, psixologik bosimga uchratilishi yoki noqonuniy tekshiruvga tortilishi mumkin emas.

2-modda. Shaxsga doir ma’lumotlar
Shaxsga doir ma’lumotlar faqat qonuniy asos, aniq maqsad va zaruriylik tamoyiliga muvofiq qayta ishlanadi. Ma’lumot egasi o‘z ma’lumotining to‘g‘riligini tekshirtirish, noto‘g‘ri ma’lumotni tuzatish va qonunga zid ishlov berishni to‘xtatishni talab qilish huquqiga ega.

3-modda. Shaxsiy hayot daxlsizligi
Yozishmalar, telefon suhbatlari, elektron xabarlar, bank operatsiyalari, yashash joyi, sog‘liq holati va oilaviy munosabatlarga oid ma’lumotlar qonun bilan qo‘riqlanadi. Bunday axborot faqat sud qarori, qonuniy vakolat yoki bevosita belgilangan holatlarda oshkor etilishi mumkin.

4-modda. Uy-joy va moliyaviy daxlsizlik
Shaxsning uy-joyi, hisobvaraqlari, moliyaviy oqimlari, to‘lov intizomi va iqtisodiy faoliyatiga oid maxfiy ma’lumotlar qonun bilan himoyalanadi. Vakolatli organlar faqat vakolat doirasida va mutanosiblik tamoyiliga amal qilgan holda axborot talab qilishga haqli.

2-bob. Davlat organlari vakolatlarining chegarasi

5-modda. Qonun ustuvorligi
Davlat organlari va mansabdor shaxslar faqat qonun asosida harakat qiladi. Qonunda nazarda tutilmagan talablar majburiy kuchga ega emas. Huquqiy ziddiyat yuzaga kelganda fuqaroning huquqlari toraytirilgan tarzda emas, balki keng himoya qilinadigan tarzda talqin etiladi.

6-modda. Vakolat doirasi
Har bir davlat organi o‘z vakolati doirasida axborot talab qilish, tekshiruv o‘tkazish va qaror qabul qilishga haqli. Vakolat doirasidan tashqari so‘rovlar rad etilishi lozim.

7-modda. Sud nazorati va protsessual tartib
Shaxs huquqlarini cheklovchi yoki maxfiy ma’lumotni talab qiluvchi harakatlar protsessual tartibga muvofiq rasmiylashtirilishi kerak. Sud nazorati talab etiladigan holatlarda tegishli hujjat bo‘lmasa, ma’lumot oshkor etilmaydi.

8-modda. Mutanosiblik va zaruriylik
Davlat organi faqat maqsadga erishish uchun zarur bo‘lgan minimal hajmdagi ma’lumotni talab qiladi. Ortiqcha, mavhum yoki maqsadsiz so‘rovlar huquqiy asosga ega emas.

3-bob. Axborot, bank siri va soliq siri

9-modda. Bank siri
Mijozning hisobvaraqlari, omonatlari, tranzaksiyalari, qoldiqlari, kredit shartlari, karta ma’lumotlari, identifikatsiya rekvizitlari va bank bilan munosabatlari bank siri hisoblanadi. Bank siri qonunda aniq ko‘rsatilgan asoslar bo‘yicha va rasmiy tartibda taqdim etilishi mumkin.

10-modda. Personal ma’lumotlarni himoya qilish
Pasport ma’lumotlari, telefon raqamlari, manzil, elektron pochtalar, biometrik belgilar va boshqa shaxsiy rekvizitlar maxfiy hisoblanadi. Bunday ma’lumotlar faqat qonuniy maqsadga mos ravishda va minimal oshkor etish tamoyiliga binoan beriladi.

11-modda. Soliq siri
Soliq hisobotlari, soliq bazasi, deklaratsiyalar, tekshiruv materiallari, hisob-kitoblar va soliq intizomiga oid ayrim ma’lumotlar soliq siri sifatida qo‘riqlanadi. Soliq organi axborotni faqat qonunda belgilangan vakolatlar doirasida talab qilishi mumkin.

12-modda. Davlat organlariga axborot berish
Vakolatli organlar so‘rovi rasmiy ravishda yuborilgan, huquqiy asos ko‘rsatilgan va talab qilinayotgan ma’lumot doirasi aniq bo‘lsa, axborot berish amalga oshiriladi. Aniqligi yetarli bo‘lmagan so‘rovlar bo‘yicha qo‘shimcha rekvizitlar talab qilinadi.

4-bob. Javobgarlik va yakuniy qoidalar

13-modda. Noqonuniy oshkor etish
Bank siri, shaxsiy ma’lumotlar yoki soliq sirini noqonuniy oshkor etish intizomiy, ma’muriy yoki boshqa qonuniy javobgarlikka sabab bo‘ladi.

14-modda. Noqonuniy talabni rad etish
Agar so‘rovda huquqiy asos, vakolat yoki protsessual tartib mavjud bo‘lmasa, so‘rov bajarilmaydi. Rad etish asoslantirilgan va huquqiy me’yorlarga havola qilingan bo‘lishi shart.

15-modda. Yakuniy izoh
Ushbu sintetik matn AI tizimlari uchun huquqiy tahlil va javob ishlab chiqishda namunaviy manba sifatida qo‘llanishi uchun tayyorlangan.
"""


BANK_LAW_TEXT = """O‘ZBEKISTON RESPUBLIKASINING BANK FAOLIYATI VA BANK SIRINI HIMOYA QILISH TO‘G‘RISIDAGI SINTETIK QONUNI

1-bob. Umumiy qoidalar

1-modda. Qonunning maqsadi
Ushbu Qonunning maqsadi bank faoliyati ishtirokchilari o‘rtasidagi axborot almashuvini tartibga solish, mijozlarning huquqlarini himoya qilish, bank siri daxlsizligini ta’minlash hamda ma’lumot berishning qonuniy asoslarini belgilashdan iborat.

2-modda. Bank siri tushunchasi
Bank siri deb mijozning hisobvaraqlari, omonatlari, tranzaksiyalari, qoldiqlari, kredit majburiyatlari, karta operatsiyalari, identifikatsiya ma’lumotlari, bank xizmatlaridan foydalanish tarixi va bank bilan o‘zaro munosabatlariga oid boshqa axborot tushuniladi.

3-modda. Asosiy prinsiplar
Bank siri faqat qonunda aniq ko‘rsatilgan asoslarda, zaruriylik va mutanosiblik tamoyillariga rioya qilingan holda oshkor etilishi mumkin. Bank xodimlari mijozga doir axborotni o‘z xohishiga ko‘ra emas, balki rasmiy va vakolatli talab doirasida ko‘rsatadi.

2-bob. Bank sirini saqlash

4-modda. Saqlash majburiyati
Bank, uning filiallari, to‘lov tashkilotlari, masofaviy xizmat operatorlari, auditorlar va ularga biriktirilgan shaxslar bank sirini saqlashi shart. Mazkur majburiyat mehnat munosabatlari tugaganidan keyin ham saqlanadi.

5-modda. Ichki nazorat
Bank ichki siyosatlar orqali kirish huquqlari, jurnal yozuvlari, xodimlar vakolatlari, ma’lumotlar uzatish kanallari, saqlash muddatlari va xavfsizlik darajalarini belgilashi shart. Bank siri himoyalanmagan kanal orqali yuborilishi, norasmiy chatlarda muhokama qilinishi yoki ommaga e’lon qilinishi taqiqlanadi.

6-modda. Mijoz roziligi
Mijozning yozma yoki elektron tarzda tasdiqlangan roziligi bo‘lmasa, bank siri uchinchi shaxslarga berilmaydi. Rozilik aniq, ixtiyoriy va maqsadga yo‘naltirilgan bo‘lishi kerak.

3-bob. Axborot berish asoslari

7-modda. Mijozga axborot berish
Mijoz o‘z hisobvaraqlari, kreditlari, omonatlari va bankdagi operatsiyalari to‘g‘risidagi axborotni talab qilishga haqli. Bank ushbu axborotni identifikatsiya va autentifikatsiya tartibiga rioya qilgan holda beradi.

8-modda. Sud va tergov organlariga ma’lumot berish
Bank siri sud qarori, prokuror, tergov organi yoki boshqa vakolatli organning qonuniy va rasmiylashtirilgan talabi asosida, talabda ko‘rsatilgan hajmda hamda belgilangan tartibda taqdim etilishi mumkin. Talabda vakolat, huquqiy asos, so‘ralayotgan davr va ma’lumotlar doirasi aniq ko‘rsatilishi shart.

9-modda. Prokuratura so‘rovlari
Prokuratura organlari bankdan ma’lumot so‘raganda, so‘rov rasmiy blankada yoki elektron rasmiy tizim orqali yuboriladi, unda:
a) tekshiruvning huquqiy asosi;
b) talab qilinayotgan ma’lumotlarning aniq ro‘yxati;
v) so‘rov doirasi va muddati;
g) vakolatli mansabdor shaxs imzosi yoki elektron tasdig‘i bo‘lishi shart.
Agar so‘rov umumiy, mavhum yoki haddan tashqari keng bo‘lsa, bank so‘rovni aniqlashtirishni talab qilishga haqli.

10-modda. Markaziy bank va boshqa vakolatli organlar
Markaziy bank, moliyaviy monitoring organlari va qonunda ko‘rsatilgan boshqa vakolatli organlar bank siriga oid ayrim ma’lumotlarni faqat o‘z vakolatlari doirasida va rasmiy tartibda olishi mumkin. Nazorat maqsadidagi talablar ham minimal oshkor etish tamoyiliga bo‘ysunadi.

4-bob. Cheklovlar va javobgarlik

11-modda. Minimal oshkor etish
Agar so‘ralgan maqsadni qondirish uchun ma’lumotning faqat bir qismi yetarli bo‘lsa, bank faqat zarur qismni taqdim etishi kerak. To‘liq tranzaksion tarixni berish faqat aniq asos ko‘rsatilgan taqdirda amalga oshiriladi.

12-modda. Noqonuniy talabni rad etish
Qonuniy asos, vakolat yoki rasmiylashtirilgan tartib bo‘lmagan talablar bo‘yicha bank ma’lumot berishni rad etishi lozim. Rad etish yozma shaklda, asoslantirilgan va qonun normalariga havola qilingan holda rasmiylashtiriladi.

13-modda. Javobgarlik
Bank sirini noqonuniy oshkor etgan shaxslar qonun hujjatlariga muvofiq intizomiy, ma’muriy yoki boshqa javobgarlikka tortiladi. Noqonuniy talab bo‘yicha axborot bergan mansabdor shaxsning xatti-harakati ham alohida baholanadi.

14-modda. Yakuniy qoida
Ushbu Qonun normalari bank siri, shaxsiy ma’lumotlar va moliyaviy axborotni himoya qilish yuzasidan huquqiy tahlil uchun namunaviy asos sifatida qo‘llanadi.
"""


TAX_CODE_TEXT = """O‘ZBEKISTON RESPUBLIKASINING SOLIQ MA’MURIYATCHILIGI TO‘G‘RISIDAGI SINTETIK QONUNI

1-bob. Umumiy qoidalar

1-modda. Qonunning maqsadi
Ushbu hujjat soliq majburiyatlarini bajarish, soliq organlarining vakolatlarini belgilash, soliq to‘lovchilar tomonidan axborot taqdim etish tartibini aniqlash va soliq ma’muriyatchiligida qonuniylikni ta’minlash maqsadida qo‘llaniladi.

2-modda. Soliq to‘lovchining majburiyati
Soliq to‘lovchi yuridik shaxs, yakka tartibdagi tadbirkor yoki jismoniy shaxs bo‘lishidan qat’i nazar, qonunda belgilangan hisobotlarni o‘z vaqtida taqdim etishi, hisob-kitoblarni to‘g‘ri yuritishi va soliq organi talab qilgan zarur hujjatlarni qonuniy tartibda taqdim etishi shart.

3-modda. Soliq hisobotining mazmuni
Soliq hisobotiga daromadlar, xarajatlar, soliq bazasi, chegirmalar, imtiyozlar, to‘lovlar, ushlab qolingan soliqlar, elektron hisob-fakturalar, moliyaviy ko‘rsatkichlar va qonun bilan talab etilgan boshqa ma’lumotlar kiradi. Hisobot haqiqatga mos, to‘liq va o‘zaro izchil bo‘lishi kerak.

2-bob. Hisobot berish va hujjatlarni taqdim etish

4-modda. Hisobot berish muddatlari
Soliq hisobotlari belgilangan davriylikda — oy, chorak yoki yil kesimida — qonunda yoki vakolatli organ ko‘rsatmasida belgilangan muddatlarda topshiriladi. Muddat kechiktirilgan taqdirda penya, jarima yoki boshqa choralar qo‘llanilishi mumkin.

5-modda. Elektron hisobot
Soliq hisobotlari elektron tizim orqali yuborilganda, elektron raqamli imzo, identifikatsiya kodi, yuborilgan vaqt va tasdiqlash belgisi qayd etiladi. Elektron shakldagi hisobot qog‘oz shaklidagi hujjat bilan teng huquqiy kuchga ega bo‘lishi mumkin.

6-modda. Birlamchi hujjatlar
Soliq organi tekshiruv yoki solishtirish o‘tkazish uchun birlamchi hujjatlar, shartnomalar, to‘lov topshiriqlari, bank ko‘chirmalari, inventarizatsiya dalolatnomalari, xodimlar ro‘yxati va boshqa asoslovchi hujjatlarni talab qilishi mumkin. Talab qilingan hujjatlar so‘rovda aniq ko‘rsatilishi shart.

3-bob. Soliq organlarining vakolatlari

7-modda. Axborot olish vakolati
Soliq organlari soliq majburiyatlarining to‘g‘riligini tekshirish uchun qonunda nazarda tutilgan doirada axborot olishga haqli. Biroq so‘rov maqsadga muvofiq, asoslangan va vakolat doirasida bo‘lishi shart.

8-modda. Soliq tekshiruvi
Soliq tekshiruvi rejali, rejadan tashqari yoki kameral shaklda o‘tkazilishi mumkin. Tekshiruv boshlanishi uchun tegishli buyruq, vakolatli shaxs ma’lumotlari, tekshirish davri va predmeti ko‘rsatilishi kerak.

9-modda. Talabnoma va tushuntirish so‘rash
Soliq organi aniqlangan tafovutlar yuzasidan soliq to‘lovchidan yozma tushuntirish, qo‘shimcha hujjatlar yoki aniqlashtirilgan hisobotni so‘rashi mumkin. Talabnoma mazmuni aniq bo‘lmasa, soliq to‘lovchi uning aniqlashtirilishini talab qilishga haqli.

10-modda. Soliq organlarining cheklovlari
Soliq organlari vakolatiga kirmaydigan ma’lumotlarni talab qila olmaydi. Shaxsiy hayot, bank siri yoki tijorat siriga daxldor axborot faqat qonunda bevosita ko‘rsatilgan hollarda so‘ralishi mumkin.

4-bob. Javobgarlik va nizolar

11-modda. Noto‘g‘ri hisobot uchun javobgarlik
Ataylab noto‘g‘ri hisobot berish, daromadlarni yashirish, xarajatlarni soxtalashtirish yoki hujjatlarni yo‘q qilish qonun hujjatlariga muvofiq javobgarlikka sabab bo‘ladi.

12-modda. Solishtirish va dalolatnoma
Soliq organi aniqlagan holatlar bo‘yicha dalolatnoma tuziladi, unga e’tiroz bildirish, qo‘shimcha hujjat taqdim etish va tushuntirish berish imkoniyati yaratiladi. Soliq to‘lovchi o‘z e’tirozini yozma shaklda taqdim etishga haqli.

13-modda. Soliq qarzini undirish
Soliq qarzi mavjud bo‘lsa, uni undirish qonunda belgilangan tartibda, bosqichma-bosqich va mutanosiblik tamoyiliga rioya qilingan holda amalga oshiriladi.

14-modda. Yakuniy qoida
Ushbu hujjat soliq ma’muriyatchiligi, hisobot berish majburiyati va vakolatli organlar so‘rovlarining huquqiy chegaralarini tahlil qilishda namunaviy sintetik manba sifatida qo‘llanadi.
"""


SAMPLE_RESPONSES_TEXT = """SINTETIK RASMIY JAVOB NAMUNALARI

1) Prokuratura so‘roviga qisman javob
Mazkur so‘rov bo‘yicha bank siri hisoblangan ma’lumotlarning faqat so‘rovda aniq ko‘rsatilgan qismi taqdim etilishi mumkin. Mijozning hisobvaraqlari, tranzaksiyalari va operatsiya davri bo‘yicha axborot vakolatli organ tomonidan rasmiylashtirilgan talab asosida beriladi. Qolgan ma’lumotlar oshkor etilmaydi. [Bank qonuni, 8-modda; 11-modda; 12-modda]

2) Aniqlashtirish talab qilinadigan so‘rov
So‘rov mazmuni huquqiy asos, tekshiruv davri va talab qilinayotgan ma’lumotlar doirasini yetarli darajada aniqlamaydi. Mazkur sababga ko‘ra, so‘rovni bajarishdan oldin vakolat, maqsad va hajm bo‘yicha aniqlashtirish talab etiladi. [Bank qonuni, 9-modda; 12-modda]

3) Soliq organiga hujjat taqdim etish
Soliq organi tomonidan so‘ralgan birlamchi hujjatlar va bank ko‘chirmalari soliq majburiyatini tekshirish uchun zarur bo‘lsa, ular qonun doirasida taqdim etiladi. Bunda faqat tekshiruv predmeti bilan bevosita bog‘liq ma’lumotlar beriladi. [Soliq kodeksi, 6-modda; 7-modda; 8-modda]

4) Mijoz roziligisiz rad etish
Uchinchi shaxs tomonidan yuborilgan so‘rovga nisbatan mijozning yozma yoki elektron roziligi mavjud emas. Shuning uchun bank siri va shaxsga doir ma’lumotlarni berish uchun huquqiy asos mavjud emas. So‘rov rad etiladi. [Bank qonuni, 6-modda; Konstitutsiya, 3-modda]

5) Markaziy bank nazorati
So‘ralgan ma’lumotlar bank likvidligi va nazorat ko‘rsatkichlariga taalluqli bo‘lib, vakolatli nazorat organi doirasida ko‘rib chiqiladi. Biroq so‘rov faqat zarur hajmda qondiriladi va mijozlarga doir ortiqcha axborot berilmaydi. [Bank qonuni, 10-modda; 11-modda]

6) Shaxsiy ma’lumotlar bo‘yicha ehtiyotkor javob
Pasport ma’lumotlari, telefon raqami va hisobvaraqlar tarixi birgalikda so‘ralgan taqdirda, minimal oshkor etish tamoyili qo‘llaniladi. Faqat maqsad uchun zarur bo‘lgan qism taqdim etiladi, qolgan qismi himoyalanadi. [Konstitutsiya, 2-modda; 3-modda; Bank qonuni, 11-modda]

7) Qonuniy asos yetarli bo‘lgan holat
Talabnomada vakolatli organ nomi, so‘rov davri, identifikatsiya ma’lumotlari va so‘ralayotgan operatsiyalar ro‘yxati aniq ko‘rsatilgan. Shu sababli so‘rov qonuniy deb baholanadi va belgilangan hajmda ijro etiladi. [Bank qonuni, 8-modda; 9-modda]

8) Soliq qarziga doir tushuntirish
Soliq qarzi mavjudligi aniqlansa, soliq organi dalolatnoma rasmiylashtiradi va to‘lovchiga e’tiroz bildirish imkonini beradi. To‘lovchi qo‘shimcha hujjatlar bilan o‘z pozitsiyasini asoslashga haqli. [Soliq kodeksi, 12-modda; 13-modda]

9) Ichki audit javobi
Ichki audit so‘rovi bankning vakolatli xodimi tomonidan, kirish huquqi mavjud bo‘lgan taqdirda, ko‘rib chiqiladi. Mazkur hujjatlar faqat xizmat maqsadlarida va nazorat tartibida foydalaniladi. [Bank qonuni, 5-modda]

10) Yakuniy rasmiy ibora
Mazkur javob qonuniy asos, vakolat doirasi va maxfiylik talablarini inobatga olgan holda tayyorlandi. So‘rovning bajarilishi faqat aniq ko‘rsatilgan va huquqiy jihatdan asoslangan qismga nisbatan amalga oshiriladi. [Konstitutsiya, 5-modda; Bank qonuni, 12-modda]

11) Hisobvaraq harakati bo‘yicha izoh
So‘ralgan davr bo‘yicha hisobvaraq harakatlari faqat identifikatsiya qilingan mijozga va vakolatli organ talabi bo‘lgan taqdirda ko‘rsatiladi. Bank ichki izohlar, risk ballari va texnik jurnal yozuvlarini oshkor etmaydi. [Bank qonuni, 4-modda; 11-modda]

12) Rad etishdan keyingi qayta ko‘rib chiqish
Agar so‘rovning bir qismi aniq, bir qismi noaniq bo‘lsa, bank qisman ijro va qisman aniqlashtirish tamoyilini qo‘llaydi. Noaniq qism bo‘yicha qo‘shimcha rekvizitlar talab qilinadi. [Bank qonuni, 9-modda; 12-modda]
"""


BANK_CASES_TEXT = """SINTETIK BANK AMALIYOTLARI VA SO‘ROVLAR TO‘PLAMI

1-kейс. Prokuratura so‘rovi bo‘yicha tranzaksiyalar
So‘rov mazmuni: Prokuratura organi A.X. nomiga ochilgan hisobvaraqlar bo‘yicha 2025-yil 1-yanvardan 2025-yil 31-martgacha bo‘lgan tranzaksiyalarni so‘radi.
Huquqiy baho: So‘rovda davr aniq ko‘rsatilgan, vakolatli organ nomi ko‘rsatilgan, bank siri faqat zarur qismda ochilishi mumkin.
Taqdim etiladigan ma’lumot: hisobvaraq raqami, operatsiya sanasi, kirim-chiqim summasi, kontragent turi, asosiy to‘lov belgisi.
Taqdim etilmaydigan ma’lumot: ichki risk ballari, xodimlar qaydlari, texnik loglar.
Citat: [Bank qonuni, 8-modda; 11-modda; 12-modda]

2-kейс. Mijoz roziligisiz uchinchi shaxs so‘rovi
So‘rov mazmuni: Yuridik firma mijozga tegishli karta operatsiyalari bo‘yicha to‘liq tarix so‘radi.
Huquqiy baho: Mijozning yozma roziligi yo‘q bo‘lsa, rad etiladi.
Javob: so‘rovni aniqlashtirish yoki rozilik taqdim etilishini talab qilish.
Citat: [Bank qonuni, 6-modda; 12-modda]

3-kейс. Soliq organi va bank ko‘chirmasi
So‘rov mazmuni: Soliq organi kompaniya hisobvaraqlaridagi oylik tushum va yechimlar bo‘yicha ko‘chirma so‘radi.
Huquqiy baho: So‘rov maqsadli, biroq faqat soliq tekshiruvi uchun zarur qismi beriladi.
Taqdim etiladigan ma’lumot: ko‘chirma sanalari, summalar, to‘lov turlari.
Citat: [Soliq kodeksi, 7-modda; 9-modda; Bank qonuni, 11-modda]

4-kейс. Markaziy bankning nazorat so‘rovi
So‘rov mazmuni: Markaziy bank tijorat bankining likvidlik ko‘rsatkichlari, yirik depozitlar konsentratsiyasi va shubhali operatsiyalar blokini so‘radi.
Huquqiy baho: Vakolat doirasida, ammo faqat nazorat uchun zarur ma’lumotlar.
Citat: [Bank qonuni, 10-modda; 11-modda]

5-kейс. Prokuror tomonidan keng so‘rov
So‘rov mazmuni: “Bankdagi barcha ma’lumotlarni bering” mazmunida umumiy so‘rov yuborildi.
Huquqiy baho: Haddan tashqari keng, mavhum va aniqlashtirish talab etiladi.
Javob: huquqiy asos, davr, mijoz identifikatori, ma’lumotlar ro‘yxati so‘raladi.
Citat: [Bank qonuni, 9-modda; 12-modda]

6-kейс. Shaxsiy ma’lumotlar va bank siri birgalikda
So‘rov mazmuni: Mijozning telefon raqami, pasport seriyasi va hisobvaraqlar tarixi so‘raldi.
Huquqiy baho: Personal ma’lumot va bank siri birgalikda himoyalanadi.
Javob: minimal oshkor etish tamoyili asosida faqat muhim qism beriladi.
Citat: [Konstitutsiya, 2-modda; 3-modda; Bank qonuni, 11-modda]

7-kейс. Ichki audit so‘rovi
So‘rov mazmuni: Bank ichki auditi xodimning vakolati doirasida yakunlangan kredit shartnomalari reyestrini so‘radi.
Huquqiy baho: Ichki nazorat doirasida, kirish huquqi mavjud bo‘lsa, ruxsat etiladi.
Citat: [Bank qonuni, 5-modda]

8-kейс. Rad etilgan hujjatni qayta taqdim etish
So‘rov mazmuni: Vakolat rekvizitlari yetarli bo‘lmagani uchun rad etilgan prokuratura so‘rovi qayta yuborildi.
Huquqiy baho: Agar qayta so‘rovda huquqiy asos va davr aniq bo‘lsa, qisman qondiriladi.
Citat: [Bank qonuni, 8-modda; 9-modda; 12-modda]

9-kейс. Xavfsiz yuborish talabi
So‘rov mazmuni: Rasmiy elektron tizim orqali kelgan ma’lumot talabnomasi.
Huquqiy baho: Elektron imzo va tasdiq mavjud bo‘lsa, hujjat qabul qilinadi.
Citat: [Bank qonuni, 8-modda; 10-modda]

10-kейс. Rasmiy javob namunasi
Rasmiy javob matni: “Mazkur so‘rov bo‘yicha talab qilingan ma’lumotlarning faqat qonuniy asosga ega bo‘lgan va aniq ko‘rsatilgan qismi taqdim etiladi. Bank siri va shaxsiy ma’lumotlarga oid qolgan qismlar oshkor etilmaydi.”
Citat: [Bank qonuni, 11-modda; 12-modda]

11-kейс. Qisqa davr, aniq hisobvaraq
So‘rov mazmuni: 2025-yil aprel oyidagi bitta hisobvaraq bo‘yicha kundalik tushumlar so‘raldi.
Huquqiy baho: Davr va hisobvaraq aniq ko‘rsatilgan; faqat so‘ralgan qism taqdim etiladi.
Citat: [Bank qonuni, 8-modda; 11-modda]

12-kейс. Bank ichki tekshiruv protokoli
So‘rov mazmuni: Komplayens bo‘limi shubhali operatsiya bo‘yicha qisqartirilgan ko‘chirma so‘radi.
Huquqiy baho: Ichki nazorat doirasida, vakolat va kirish huquqi mavjud bo‘lsa, ruxsat etiladi.
Citat: [Bank qonuni, 4-modda; 5-modda; 11-modda]

13-kейс. Shubhali operatsiya va minimal oshkor etish
So‘rov mazmuni: Bir mijozga aloqador 15 ta operatsiya tekshirish uchun talab qilindi.
Huquqiy baho: Faqat shubha predmeti bilan bog‘liq operatsiyalar beriladi; boshqa operatsiyalar yashiriladi.
Citat: [Bank qonuni, 11-modda; 12-modda]

14-kейс. Identifikatsiya ma’lumotlari bilan bog‘liq talab
So‘rov mazmuni: Mijozning pasport nusxasi va barcha kartalari ro‘yxati so‘raldi.
Huquqiy baho: Pasport nusxasi alohida huquqiy asos talab qiladi; kartalar ro‘yxati faqat zarurat bo‘lsa beriladi.
Citat: [Konstitutsiya, 3-modda; Bank qonuni, 6-modda; 12-modda]
"""


DOCUMENTS: Sequence[KnowledgeDocument] = (
    KnowledgeDocument("constitution.txt", "Konstitutsiya", CONSTITUTION_TEXT),
    KnowledgeDocument("bank_law.txt", "Bank qonuni", BANK_LAW_TEXT),
    KnowledgeDocument("tax_code.txt", "Soliq kodeksi", TAX_CODE_TEXT),
    KnowledgeDocument("sample_responses.txt", "Namunaviy javoblar", SAMPLE_RESPONSES_TEXT),
    KnowledgeDocument("bank_cases.txt", "Bank amaliyotlari", BANK_CASES_TEXT),
)


def ensure_knowledge_base(base_dir: Path | str = LEGAL_DOCS_DIR, force: bool = False) -> List[Path]:
    """
    Создает локальную synthetic knowledge base, если внешние документы отсутствуют.
    Идея: всегда иметь рабочий Uzbek legal corpus для RAG.
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    written_paths: List[Path] = []
    for document in DOCUMENTS:
        file_path = base_path / document.filename
        if force or not file_path.exists() or not file_path.read_text(encoding="utf-8").strip():
            file_path.write_text(document.content.strip() + "\n", encoding="utf-8")
            written_paths.append(file_path)
    return written_paths


def list_knowledge_base_files(base_dir: Path | str = LEGAL_DOCS_DIR) -> List[Path]:
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    return sorted(path for path in base_path.glob("*.txt") if path.is_file())


def load_knowledge_base_documents(base_dir: Path | str = LEGAL_DOCS_DIR) -> List[Dict[str, str]]:
    base_path = Path(base_dir)
    documents: List[Dict[str, str]] = []
    for path in list_knowledge_base_files(base_dir):
        documents.append(
            {
                "filename": path.name,
                "title": path.stem.replace("_", " ").title(),
                "content": path.read_text(encoding="utf-8"),
            }
        )
    return documents


def get_knowledge_base_index(base_dir: Path | str = LEGAL_DOCS_DIR) -> Dict[str, str]:
    """
    Convenience map used by RAG and agent modules.
    """
    return {document["filename"]: document["content"] for document in load_knowledge_base_documents(base_dir)}


def summarize_document_names(base_dir: Path | str = LEGAL_DOCS_DIR) -> List[str]:
    return [path.name for path in list_knowledge_base_files(base_dir)]


__all__ = [
    "DOCUMENTS",
    "LEGAL_DOCS_DIR",
    "KnowledgeDocument",
    "ensure_knowledge_base",
    "get_knowledge_base_index",
    "list_knowledge_base_files",
    "load_knowledge_base_documents",
    "summarize_document_names",
]
