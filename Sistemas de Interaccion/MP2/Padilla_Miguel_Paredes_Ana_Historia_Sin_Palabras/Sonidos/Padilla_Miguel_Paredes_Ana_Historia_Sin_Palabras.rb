# ============================================================
# HISTORIA SONORA COMPLETA
# ESCENA 1 + ESCENA 2 + ESCENA 3 + ESCENA 4
# ============================================================

use_bpm 60


# ============================================================
# RUTAS DE AUDIO
# ============================================================

base_dir = "C:/Users/Usuario/OneDrive - PUJ Cali/Escritorio/SEMESTRES/SEMESTRE7/Sistemas de Interaccion/MP2/Padilla_Miguel_Paredes_Ana_Historia_Sin_Palabras/Padilla_Miguel_Paredes_Ana_Historia_Sin_Palabras/Sonidos/"

ruta1 = base_dir + "Escena1/"
ruta2 = base_dir + "Escena2/"
ruta3 = base_dir + "Escena3_4/"


# ============================================================
# 🎬 ESCENA 1 — EL DESPERTAR Y LA CASA
# ============================================================


# ------------------------------------------------------------
# 1. DESPERTADOR
# ------------------------------------------------------------

alarma = sample ruta1 + "alarma_despertador.wav",
  amp: 1.5,
  pan: 0

sleep 2.8

sample ruta1 + "golpe-al-despertador.wav",
  amp: 1.5,
  pan: 0

sleep 1

control alarma,
  amp: 0,
  amp_slide: 0

sleep 1


# ------------------------------------------------------------
# 2. PASOS CON PANTUFLAS
# ------------------------------------------------------------

pasos1 = sample ruta1 + "pasos_chanclas.wav",
  amp: 3,
  pan: -0.3

sleep 1.8

pasos2 = sample ruta1 + "pasos_chanclas.wav",
  amp: 4,
  pan: 0.2

sleep 1.5

control pasos1,
  amp: 0,
  amp_slide: 0.1

control pasos2,
  amp: 0,
  amp_slide: 0.1

sleep 0.15


# ------------------------------------------------------------
# 3. ABRE EL GRIFO
# ------------------------------------------------------------

sample ruta1 + "grifo.wav",
  amp: 1.2,
  pan: 0.25

sleep 1


# ------------------------------------------------------------
# 4. GOTEO / NO SALE AGUA
# ------------------------------------------------------------

goteo = sample ruta1 + "goteo.wav",
  amp: 1.0,
  pan: 0.25,
  rate: 0.5

sleep 3


# ------------------------------------------------------------
# 5. SILENCIO INCÓMODO
# ------------------------------------------------------------

sleep 1


# ------------------------------------------------------------
# 6. RESOPLIDO DE FRUSTRACIÓN
# ------------------------------------------------------------

suspiro = ruta1 + "suspiro.wav"

sample suspiro,
  amp: 1.4,
  pan: 0

sleep sample_duration suspiro


# ------------------------------------------------------------
# 7. ENCIENDE LA LUZ
# ------------------------------------------------------------

sample ruta1 + "interruptor_bombilla.wav",
  amp: 1.2,
  pan: 0

sleep 0.4


# ------------------------------------------------------------
# 8. ZUMBIDO ELÉCTRICO
# ------------------------------------------------------------

zumbido = sample ruta1 + "electricidad_sonido.wav",
  amp: 0.45,
  pan: 0

sleep 0.8

control zumbido,
  amp: 0.75,
  amp_slide: 0.5

sleep 0.8

control zumbido,
  amp: 1.0,
  amp_slide: 0.5

sleep 1


# ------------------------------------------------------------
# 9. BOMBILLO EXPLOTA
# ------------------------------------------------------------

control zumbido,
  amp: 0,
  amp_slide: 0.03

sleep 0.05

sample ruta1 + "bombilla_rompiendose.wav",
  amp: 1.7,
  pan: 0

sample ruta1 + "vidrio_al_suelo.wav",
  amp: 1.4,
  pan: 0

sleep 0.3


# ------------------------------------------------------------
# 10. JADEO Y RESPIRACIÓN AGITADA; PUÑO EN LA PARED
# ------------------------------------------------------------

gasp1 = ruta1 + "gasp.wav"

sample gasp1,
  amp: 1.5,
  pan: 0

sample ruta1 + "respiracion_agitada.wav",
  amp: 1.8,
  finish: 0.35

sleep 1.2

sample ruta1 + "golpe_fuerte_pared.wav",
  amp: 1.8

sleep 1.0


# ============================================================
# FIN ESCENA 1
# PAUSA ENTRE ESCENAS
# ============================================================

sleep 5


# ============================================================
# 🎬 ESCENA 2 — EL GARAJE Y LA CALLE
# ============================================================


# ------------------------------------------------------------
# 11. PUERTA METÁLICA DEL GARAJE
# ------------------------------------------------------------

puerta_garaje = ruta2 + "puerta_metal.wav"

sample puerta_garaje,
  amp: 1.5,
  pan: 0

sleep sample_duration puerta_garaje

sleep 0.2


# ------------------------------------------------------------
# 12. PUERTA DEL CARRO
# ------------------------------------------------------------

puerta_carro = ruta2 + "puerta_carro.wav"

sample puerta_carro,
  amp: 1.4,
  pan: 0.2

sleep sample_duration puerta_carro

sleep 0.2


# ------------------------------------------------------------
# 13. LLAVE EN EL ARRANQUE
# ------------------------------------------------------------

llaves = ruta2 + "llaves.wav"

sample llaves,
  amp: 1.3,
  pan: 0.1

sleep sample_duration llaves

sleep 0.2


# ------------------------------------------------------------
# 14. MOTOR INTENTANDO ARRANCAR
# ------------------------------------------------------------

# Este archivo ya contiene los tres intentos

auto = ruta2 + "auto_arrancando.wav"

sample auto,
  amp: 1.5,
  pan: 0

sleep sample_duration auto

sleep 0.5


# ------------------------------------------------------------
# 15. GOLPES AL VOLANTE
# ------------------------------------------------------------

golpes = sample ruta2 + "golpes_volante.wav",
  amp: 1.6,
  pan: 0,
  start: 0.2

sleep 5

control golpes,
  amp: 0,
  amp_slide: 0.05

sleep 0.1


# ------------------------------------------------------------
# 16. PORTAZO DEL CARRO
# ------------------------------------------------------------

sample ruta2 + "puerta_carro_saliendo.wav",
  amp: 1.7,
  pan: 0

sleep 2.2


# ------------------------------------------------------------
# 17. PASOS RÁPIDOS
# ------------------------------------------------------------

sample ruta2 + "pasos.wav",
  amp: 1.5,
  pan: -0.2

sleep 1.5


# ------------------------------------------------------------
# 18. AMBIENTE DE LLUVIA
# ------------------------------------------------------------

lluvia = sample ruta2 + "lluvia.wav",
  amp: 0.65,
  pan: 0,
  finish: 0.12

sleep 2


# ------------------------------------------------------------
# 19. MOTOR ACERCÁNDOSE
# ------------------------------------------------------------

carro = sample ruta2 + "carro_acercandose.wav",
  amp: 0.20,
  pan: -0.4,
  finish: 0.65

sleep 1.5

control carro,
  amp: 0.5,
  amp_slide: 1.0,
  pan: -0.2

sleep 1.2

control carro,
  amp: 0.9,
  amp_slide: 0.8,
  pan: -0.1

sleep 1.0

control carro,
  amp: 1.3,
  amp_slide: 0.5,
  pan: 0

sleep 0.5


# ------------------------------------------------------------
# 20. CARRO PASANDO POR EL CHARCO
# ------------------------------------------------------------

salpicadura = ruta2 + "salpicadura_agua.wav"

sample salpicadura,
  amp: 1.9,
  pan: 0

sleep sample_duration salpicadura


# ------------------------------------------------------------
# 21. REACCIÓN / JADEO
# ------------------------------------------------------------

gasp2 = ruta2 + "gasp.wav"

sample gasp2,
  amp: 4,
  pan: 0

sleep sample_duration gasp2

sleep 0.2


# ------------------------------------------------------------
# 22. GRUÑIDO / REACCIÓN ENOJADA
# ------------------------------------------------------------

grunido = ruta2 + "grunido.wav"

sample grunido,
  amp: 2.0,
  pan: 0

sleep sample_duration grunido

sleep 0.15


# ------------------------------------------------------------
# 23. VOZ ENOJADA
# ------------------------------------------------------------

voz = ruta2 + "voz.wav"

sample voz,
  amp: 1.8,
  pan: 0,
  rate: 1.05

sleep sample_duration voz

sleep 0.3


# ------------------------------------------------------------
# 24. CARRO ALEJÁNDOSE
# ------------------------------------------------------------

control carro,
  amp: 0.7,
  amp_slide: 1.0,
  pan: 0.2

sleep 1.2

control carro,
  amp: 0.35,
  amp_slide: 1.2,
  pan: 0.4

sleep 1.3

control carro,
  amp: 0.12,
  amp_slide: 1.5,
  pan: 0.5

sleep 1.5


# ------------------------------------------------------------
# 25. FINAL ESCENA 2
# ------------------------------------------------------------

# El carro desaparece
control carro,
  amp: 0,
  amp_slide: 0.8,
  pan: 0.5

# La lluvia desaparece progresivamente
control lluvia,
  amp: 0,
  amp_slide: 1.5

sleep 1.6


# ============================================================
# 🎬 ESCENA 3 — LA OFICINA
# ============================================================


# ------------------------------------------------------------
# 26. AMBIENTE DE OFICINA Y TECLEO
# ------------------------------------------------------------

amb_office = sample ruta3 + "Office-Ambience.wav",
  amp: 0.6

amb_keys = sample ruta3 + "keyboard-typing.wav",
  amp: 0.4

sleep 1.0


# ------------------------------------------------------------
# 27. ENTRADA CON ZAPATOS EMPAPADOS
# ------------------------------------------------------------

squeak = sample ruta3 + "Shoe-Squeaks.wav",
  start: 0.122,
  finish: 0.245,
  amp: 1.6

sleep 2.8

control squeak,
  amp: 0,
  amp_slide: 0.2


# ------------------------------------------------------------
# 28. PORTAZO DEL DESPACHO
# ------------------------------------------------------------

kill amb_keys

sample ruta3 + "Wood-Impact.wav",
  amp: 1.8

sleep 0.5


# ------------------------------------------------------------
# 29. GRITOS ININTELIGIBLES DEL JEFE
# ------------------------------------------------------------

# El audio total dura aproximadamente 17.8 s.
# La parte de furia comienza alrededor del segundo 8.

sample ruta3 + "Angry-man-shouting.wav",
  start: 0.45,
  finish: 0.65,
  amp: 2.3

sleep 3.6

# ------------------------------------------------------------
# 30. MANOTAZO DEL JEFE EN EL ESCRITORIO
# ------------------------------------------------------------

sample ruta3 + "community-table-slam.wav", amp: 2.2
sample ruta3 + "Wood-Impact.wav", amp: 1.5
sleep 0.6

# ------------------------------------------------------------
# 31. LLANTO Y REDUCCIÓN DE OFICINA
# ------------------------------------------------------------

control amb_office,
  amp: 0.15,
  amp_slide: 0.8

crying = sample ruta3 + "man-crying.wav",
  amp: 1.8

sleep 1.4


# ------------------------------------------------------------
# 31. EMPACA SUS COSAS
# ------------------------------------------------------------

clatter = sample ruta3 + "Desk stationery clatter.wav",
  start: 0.055,
  amp: 2.5,
  rate: 1.35

sleep 4.2

control clatter,
  amp: 0,
  amp_slide: 0.3

control crying,
  amp: 0,
  amp_slide: 1.0


# ------------------------------------------------------------
# 32. PASOS APRESURADOS
# ------------------------------------------------------------

sample ruta3 + "footsteps-in-a-hallway.wav",
  amp: 1.4

sleep 2.4


# ------------------------------------------------------------
# 33. PUERTA METÁLICA CORTAFUEGOS
# ------------------------------------------------------------

sample ruta3 + "Heavy-metal-door-slam.wav",
  start: 0.38,
  amp: 2.2

kill amb_office


# ============================================================
# FIN ESCENA 3
# PAUSA ENTRE ESCENA 3 Y ESCENA 4
# ============================================================

sleep 5.5


# ============================================================
# 🎬 ESCENA 4 — EL ASALTO Y DESENLACE
# ============================================================

with_fx :lpf,
  cutoff: 125,
cutoff_slide: 0.2 do |rain_filter|
  
  
  # ----------------------------------------------------------
  # 34. LLUVIA EXTERIOR
  # ----------------------------------------------------------
  
  rain_track = sample ruta3 + "lluvia-rain.wav",
    amp: 0.2,
    sustain: 50,
    release: 5,
    rate: 0.95
  
  control rain_track,
    amp: 0.55,
    amp_slide: 3.5
  
  sleep 4.5
  
  
  # ----------------------------------------------------------
  # 35. GRITOS AGRESIVOS
  # ----------------------------------------------------------
  
  sample ruta3 + "Aggressive shouting.wav",
    amp: 1.8
  
  sleep 2.2
  
  
  # ----------------------------------------------------------
  # 36. EMPUJÓN / CAÍDA DE LA CAJA
  # ----------------------------------------------------------
  
  sample ruta3 + "cardboard-box-drop-hit.wav",
    amp: 2.0,
    finish: 0.08
  
  sleep 0.15
  
  sample ruta3 + "multi-layered-cashing-noise.wav",
    amp: 1.0
  
  sleep 1.8
  
  
  # ----------------------------------------------------------
  # 37. AMARTILLADO DEL ARMA
  # ----------------------------------------------------------
  
  sample ruta3 + "Gun cocking mechanical.wav",
    amp: 1.6
  
  sleep 1.2
  
  
  # ----------------------------------------------------------
  # 38. DISPARO
  # ----------------------------------------------------------
  
  sample ruta3 + "Gunshot echo.wav",
    amp: 2.5
  
  
  # ----------------------------------------------------------
  # 39. IMPACTO ACÚSTICO
  # ----------------------------------------------------------
  
  control rain_filter,
    cutoff: 62,
    cutoff_slide: 0.05
  
  
  # ----------------------------------------------------------
  # 40. TINNITUS + ASALTANTES HUYENDO
  # ----------------------------------------------------------
  
  tin = sample ruta3 + "Tinnitus ringing high pitch.wav",
    amp: 4.0,
    attack: 0.05,
    sustain: 8,
    release: 2,
    rate: 0.75
  
  sample ruta3 + "Running-shoes-wet-pavement.wav",
    amp: 0.55,
    rate: 1.15
  
  sleep 1.2
  
  
  # ----------------------------------------------------------
  # 41. TAQUICARDIA + RESPIRACIÓN
  # ----------------------------------------------------------
  
  sample ruta3 + "rapid-heartbeat.wav",
    amp: 1.8
  
  sample ruta3 + "heavy-breathing-type-01.wav",
    start: 0.14,
    amp: 1.6
  
  sleep 3.4
  
  
  # ----------------------------------------------------------
  # 42. ÚLTIMOS LATIDOS
  # ----------------------------------------------------------
  
  sample ruta3 + "heartbeat.wav",
    amp: 2.0,
    rate: 0.75,
    finish: 0.35
  
  sleep 2.4
  
  
  # ----------------------------------------------------------
  # 43. CUERPO CAE AL PAVIMENTO
  # ----------------------------------------------------------
  
  sample ruta3 + "Body-fall.wav",
    start: 0.36,
    finish: 0.58,
    amp: 2.2
  
  sleep 2.2
  
  
  # ----------------------------------------------------------
  # 44. LLUVIA SE INTENSIFICA
  # ----------------------------------------------------------
  
  control tin,
    amp: 0,
    amp_slide: 2.0
  
  control rain_filter,
    cutoff: 128,
    cutoff_slide: 1.5
  
  control rain_track,
    amp: 1.45,
    amp_slide: 2.5
  
  sleep 4.5
  
  
  # ----------------------------------------------------------
  # 45. FADE OUT FINAL
  # ----------------------------------------------------------
  
  control rain_track,
    amp: 0,
    amp_slide: 4.0
  
  sleep 4.5
  
end


# ============================================================
# FIN DE LA HISTORIA
# SILENCIO ABSOLUTO
# ============================================================