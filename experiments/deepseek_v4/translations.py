"""Translation table for eval intermediates.

The lens surfaces concepts in whichever language the model represents them in
(measured: 32.9% of J-lens's top-10 is non-English vs 3.9% for vanilla), so
English-only targets systematically understate it. This covers the concept
FAMILIES that make up most instances rather than 444 words individually.

Chinese first (the dominant non-English readout on this model), then the
Romance/German forms that matter for the multilingual set.
"""

_NUM = {
    "0": ["零", "cero", "zéro", "null"],
    "1": ["一", "uno", "un", "eins", "um"],
    "2": ["二", "两", "dos", "deux", "zwei", "dois"],
    "3": ["三", "tres", "trois", "drei", "três"],
    "4": ["四", "cuatro", "quatre", "vier"],
    "5": ["五", "cinco", "cinq", "fünf"],
    "6": ["六", "seis", "sechs"],
    "7": ["七", "siete", "sept", "sieben"],
    "8": ["八", "ocho", "huit", "acht", "oito"],
    "9": ["九", "nueve", "neuf", "neun", "nove"],
    "10": ["十", "diez", "dix", "zehn", "dez"],
    "11": ["十一", "once", "onze", "elf"],
    "12": ["十二", "doce", "douze", "zwölf"],
    "13": ["十三", "trece", "treize"],
    "14": ["十四", "catorce", "quatorze"],
    "15": ["十五", "quince", "quinze"],
    "16": ["十六", "dieciséis", "seize"],
    "18": ["十八", "dieciocho", "dix-huit"],
    "20": ["二十", "veinte", "vingt", "zwanzig"],
    "24": ["二十四"], "25": ["二十五"], "30": ["三十", "treinta", "trente"],
    "36": ["三十六"], "40": ["四十", "cuarenta"], "45": ["四十五"],
    "50": ["五十", "cincuenta"], "60": ["六十"], "64": ["六十四"],
    "100": ["一百", "cien", "cent", "hundert"],
}
_WORDNUM = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "eighteen": "18", "twenty": "20",
    "thirty": "30", "forty": "40", "fifty": "50", "hundred": "100",
}
_MONTH = {
    "January": ["一月", "enero", "janvier", "Januar"],
    "February": ["二月", "febrero", "février", "Februar"],
    "March": ["三月", "marzo", "mars", "März"],
    "April": ["四月", "abril", "avril"],
    "May": ["五月", "mayo", "mai", "Mai"],
    "June": ["六月", "junio", "juin", "Juni"],
    "July": ["七月", "julio", "juillet", "Juli"],
    "August": ["八月", "agosto", "août"],
    "September": ["九月", "septiembre", "septembre"],
    "October": ["十月", "octubre", "octobre", "Oktober"],
    "November": ["十一月", "noviembre", "novembre"],
    "December": ["十二月", "diciembre", "décembre", "Dezember"],
}
_OTHER = {
    # operations
    "addition": ["加", "加法", "+", "suma", "plus"],
    "subtraction": ["减", "减法", "-", "resta", "minus"],
    "multiplication": ["乘", "乘法", "*", "×", "multiplicación"],
    "division": ["除", "除法", "/", "÷", "división"],
    "squared": ["平方", "²", "cuadrado"],
    # colors
    "red": ["红", "红色", "rojo", "rouge", "rot", "vermelho"],
    "blue": ["蓝", "蓝色", "azul", "bleu", "blau"],
    "green": ["绿", "绿色", "verde", "vert", "grün"],
    "black": ["黑", "黑色", "negro", "noir", "schwarz", "preto"],
    "white": ["白", "白色", "blanco", "blanc", "weiß"],
    "yellow": ["黄", "黄色", "amarillo", "jaune", "gelb"],
    # size / position / age
    "big": ["大", "grande", "grand", "groß"],
    "small": ["小", "pequeño", "petit", "klein", "pequeno"],
    "old": ["老", "旧", "viejo", "vieux", "alt", "velho"],
    "young": ["年轻", "joven", "jeune", "jung"],
    "near": ["近", "cerca", "près", "nah"],
    "far": ["远", "lejos", "loin", "fern"],
    "up": ["上", "arriba", "haut", "oben"],
    "down": ["下", "abajo", "bas", "unten"],
    "first": ["第一", "primero", "premier", "erste"],
    "last": ["最后", "último", "dernier", "letzte"],
    "half": ["一半", "半", "mitad", "moitié", "Hälfte"],
    "double": ["双", "两倍", "doble", "double"],
    "light": ["光", "轻", "luz", "lumière", "Licht"],
    "dark": ["暗", "黑暗", "oscuro", "sombre", "dunkel"],
    "hot": ["热", "caliente", "chaud", "heiß"],
    "cold": ["冷", "frío", "froid", "kalt"],
    "fast": ["快", "rápido", "rapide", "schnell"],
    "slow": ["慢", "lento", "lent", "langsam"],
    # nouns
    "opposite": ["相反", "反义", "对立", "contrario", "contraire", "Gegenteil"],
    "color": ["颜色", "色", "color", "couleur", "Farbe", "cor"],
    "month": ["月份", "月", "mes", "mois", "Monat"],
    "number": ["数字", "数", "número", "nombre", "Zahl"],
    "time": ["时间", "时", "tiempo", "temps", "Zeit"],
    "place": ["地方", "地点", "lugar", "lieu", "Ort"],
    "day": ["日", "天", "día", "jour", "Tag", "dia"],
    "night": ["夜", "晚", "夜晚", "noche", "nuit", "Nacht", "noite"],
    "season": ["季节", "季", "estación", "saison", "Jahreszeit"],
    "summer": ["夏", "夏天", "verano", "été", "Sommer"],
    "winter": ["冬", "冬天", "invierno", "hiver", "Winter"],
    "family": ["家庭", "家", "familia", "famille", "Familie"],
    "size": ["大小", "尺寸", "tamaño", "taille", "Größe"],
    "order": ["顺序", "次序", "orden", "ordre"],
    "water": ["水", "agua", "eau", "Wasser", "água"],
    "death": ["死", "死亡", "muerte", "mort", "Tod"],
    "blood": ["血", "sangre", "sang", "Blut"],
    "body": ["身体", "cuerpo", "corps", "Körper"],
    "fear": ["恐惧", "害怕", "miedo", "peur", "Angst"],
    # languages / countries
    "German": ["德语", "德国", "Deutsch", "alemán", "allemand"],
    "French": ["法语", "法国", "français", "francés"],
    "Spanish": ["西班牙语", "西班牙", "español", "espagnol"],
    "Italian": ["意大利语", "意大利", "italiano", "italien"],
    "Portuguese": ["葡萄牙语", "葡萄牙", "português"],
    "English": ["英语", "英文", "inglés", "anglais", "Englisch"],
    "Chinese": ["中文", "汉语", "中国话"],
    "Japanese": ["日语", "日文"],
    "China": ["中国", "中華"],
    "Japan": ["日本"],
    "Brazil": ["巴西", "Brasil"],
    "Italy": ["意大利", "Italia", "Italie"],
    "France": ["法国", "Francia"],
    "Germany": ["德国", "Alemania", "Allemagne"],
    "Spain": ["西班牙", "España"],
    "Russia": ["俄罗斯", "Rusia"],
    "Egypt": ["埃及"],
    "India": ["印度"],
    "euro": ["欧元", "€"],
    "dollar": ["美元", "$"],
    "yen": ["日元"],
    "pound": ["英镑", "£"],
    "español": ["西班牙语", "Spanish"],
    "português": ["葡萄牙语", "Portuguese"],
}


def variants(word: str) -> list[str]:
    """Non-English forms of `word`, or [] if we have no confident mapping."""
    out = []
    if word in _OTHER:
        out += _OTHER[word]
    if word in _NUM:
        out += _NUM[word]
    lw = word.lower()
    if lw in _WORDNUM:                       # "three" -> digit + its translations
        d = _WORDNUM[lw]
        out += [d] + _NUM.get(d, [])
    if word in _MONTH:
        out += _MONTH[word]
    if lw in _OTHER:
        out += _OTHER[lw]
    return out


def coverage(words) -> float:
    words = list(words)
    return sum(1 for w in words if variants(w)) / max(len(words), 1)
