import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import warnings

# Sayısal uyarıları (örn. sonsuza bölme) yakalamak için
warnings.filterwarnings('ignore', category=RuntimeWarning) # Hataları görmezden gel

# --- MODEL PARAMETRELERİ VE FİZİKSEL SABİTLER ---
RHO_MAX_PLANCK = 1.0e17  # Temsili Planck Yoğunluğu (Bkz. Şekil 3)
G_N = 1.0                # Kütleçekim sabiti (Normalize)
K_EOS = 100.0            # Hal Denklemi Sabiti
GAMMA_EOS = 2.0          # Hal Denklemi Üssü

# --- HAL DENKLEMLERİ (EQUATION OF STATE) ---

def equation_of_state(rho, model_type):
    """
    Verilen yoğunluğa (rho) karşılık gelen basıncı (P) 
    ve "sertliği" (cs_sq = dP/drho) hesaplar.
    """
    
    # Standart Madde Basıncı ve Sertliği
    P_matter = K_EOS * rho**GAMMA_EOS
    cs_sq_matter = K_EOS * GAMMA_EOS * rho**(GAMMA_EOS - 1.0)
    
    if model_type == 'GR':
        return P_matter, cs_sq_matter
    
    elif model_type == 'FTE':
        stabilization_term = (1.0 - (rho / RHO_MAX_PLANCK)**2)
        
        if stabilization_term <= 1e-9 or rho >= RHO_MAX_PLANCK:
            # Planck Çekirdeğine ulaşıldı. Sertlik sonsuz.
            return P_matter, 1e100  # Çok büyük "sertlik"
        else:
            P_eff = P_matter 
            cs_sq_fte = cs_sq_matter / stabilization_term
            return P_eff, cs_sq_fte

# --- MODİFİYE TOV DENKLEMLERİ (BÖLÜM V) ---

def tov_equations(r, y, model_type):
    """
    Modifiye Tolman-Oppenheimer-Volkoff (TOV) denklemlerinin
    diferansiyel sistemini tanımlar.
    """
    try:
        M, rho = y
        
        if rho <= 1e-10 or np.isnan(rho):
            return np.array([0.0, 0.0])
            
        P_eff, cs_sq = equation_of_state(rho, model_type)
        
        rho_eff = rho 
        G_eff = G_N
        
        if cs_sq <= 0 or np.isnan(cs_sq) or cs_sq > 1e99:
            dMdr = 4.0 * np.pi * r**2 * rho_eff
            drhodr = 0.0
            return np.array([dMdr, drhodr])

        denominator = r * (r - 2.0 * G_eff * M)
        if denominator <= 1e-20:
            # Tekillik (Olay Ufku) - GR burada çöker
            return np.array([0.0, 0.0]) 

        # Denklemler
        dMdr = 4.0 * np.pi * r**2 * rho_eff
        dPdr = - (G_eff * (rho_eff + P_eff) * (M + 4.0 * np.pi * r**3 * P_eff)) / denominator
        drhodr = dPdr / cs_sq
        
        if np.isnan(dMdr) or np.isnan(drhodr):
            return np.array([0.0, 0.0])
            
        return np.array([dMdr, drhodr])

    except RuntimeWarning:
        return np.array([0.0, 0.0])

# --- SİMÜLASYONU ÇALIŞTIRMA ---
r_start = 1e-6
r_end = 5.0  # km
r_span = [r_start, r_end]
r_points = np.linspace(r_start, r_end, 200)

central_density = 2e17  # RHO_MAX_PLANCK'tan yüksek
central_mass = 0.0

# --- MODEL 1: STANDART GR (TEKİLLİK) ---
y0_gr = [central_mass, central_density]
sol_gr = solve_ivp(
    tov_equations, 
    r_span, 
    y0_gr, 
    method='Radau',  
    t_eval=r_points, 
    args=('GR',)
)
print("Standart GR çözümü (beklendiği gibi) nümerik hatayla durdu.")

# --- MODEL 2: FTE MODELİ (PLANCK ÇEKİRDEĞİ) ---
y0_fte = [central_mass, central_density]
sol_fte = solve_ivp(
    tov_equations, 
    r_span, 
    y0_fte, 
    method='Radau', 
    t_eval=r_points, 
    args=('FTE',)
)
print("FTE Modeli (Planck Çekirdeği) çözümü başarıyla tamamlandı.")

# --- SONUÇLARI GÖRSELLEŞTİRME (ŞEKİL 3) ---
plt.figure(figsize=(10, 7))

# --- DÜZELTME: Hata kontrolü eklendi ---
# sol_gr (GR) çözümünün başarısız olmasını bekliyoruz.
# Bu nedenle, 'isinstance' ile 'ndarray' olup olmadığını ve
# 'shape[1]' (veri noktası sayısı) > 0 olup olmadığını kontrol ediyoruz.
if isinstance(sol_gr.y, np.ndarray) and sol_gr.y.shape[1] > 0:
    plt.plot(sol_gr.t, sol_gr.y[1], 'r--', label='Standart GR (Tekillik)', linewidth=2.5)
else:
    print("GR Çözümü (Tekillik) çizilemedi - beklendiği gibi nümerik olarak başarısız oldu.")

# sol_fte (FTE) çözümünün başarılı olmasını bekliyoruz.
if isinstance(sol_fte.y, np.ndarray) and sol_fte.y.shape[1] > 0:
    plt.plot(sol_fte.t, sol_fte.y[1], 'b-', label='FTE Modeli (Planck Çekirdeği)', linewidth=2.5)
else:
    print("FTE Çözümü (Planck Çekirdeği) başarısız oldu - beklenmedik hata!")

# Planck Yoğunluğu çizgisi
plt.axhline(RHO_MAX_PLANCK, color='k', linestyle=':', label=r'$\rho_{max} \approx M_{Pl}^{4}$')

# Grafik Ayarları
plt.yscale('log')
plt.ylim(1e14, 1e19) 
plt.xlim(0, 5)     
plt.title('Tekillik Çözümü (Modifiye TOV Nümerik Sonuçları)', fontsize=14)
plt.xlabel('Yarıçap (r) [km]', fontsize=12)
plt.ylabel('Yoğunluk (ρ) [log ölçek]', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--')
plt.show()

# Çıktı Kontrolleri
if isinstance(sol_fte.y, np.ndarray) and sol_fte.y.shape[1] > 0:
    print(f"FTE Modeli merkez yoğunluğu: {sol_fte.y[1][0]:.2e}")
    print(f"FTE Modeli, yoğunluğu ~{RHO_MAX_PLANCK:.1e} seviyesinde stabilize etti.")
if isinstance(sol_gr.y, np.ndarray) and sol_gr.y.shape[1] > 0:
    print(f"GR Modeli merkez yoğunluğu: {sol_gr.y[1][0]:.2e}")
