# ============================================================

# WAKE ME UP - Avicii | Guitarra + Piano + Percusión
# Miguel Angel Padilla Rosero y Ana Daniela Paredes Tovar

# ============================================================

use_bpm 106

# Definición de los acordes para la guitarra

bm = [:b2, :fs3, :b3, :d4, :fs4]
g  = [:g2, :b2, :d3, :g3, :b3, :g4]
d  = [:d3, :a3, :d4, :fs4]
a  = [:a2, :e3, :a3, :cs4, :e4]


# Funciones para simplificar la escritura de notas y sus parámetros

# Función para acordes de guitarra
define :gt do |notes, duration=0.427, acc=1.0|
  use_synth :pluck
  play_chord notes, pluck_decay: 2.137, coef: 0.3, amp: acc, release: 1.71
  sleep duration
end

# Función para notas de piano
define :p do |note, duration, rel=0.6, val_amp=0.6|
  play note, release: rel, amp: val_amp
  sleep duration
end


# Ritmo de la guitarra
# Sigue el formato (acorde, sleep, amp)

# Intro de guitarra
define :guitar_intro do
  with_fx :reverb, room: 0.5, mix: 0.25 do
    2.times do
      gt bm, 0.641, 1.3; gt bm, 0.214, 0.9; gt bm; gt bm
      gt g,  0.641, 1.3; gt g,  0.214, 0.9; gt g,  0.427, 1.1; gt g
      gt d,  0.641, 1.3; gt d,  0.214, 0.9; gt d,  0.427, 1.1; gt d
      gt a,  0.641, 1.3; gt a,  0.214, 0.9; gt a,  0.427, 1.1; gt a
    end
  end
end

# Ritmo principal de la guitarra
define :guitar_main_rhythm do
  with_fx :reverb, room: 0.5, mix: 0.25 do
    17.times do
      gt bm, 0.641, 1.3; gt bm, 0.214, 0.9; gt bm, 0.214; gt bm, 0.214, 0.9; gt bm, 0.427, 1.1
      gt g,  0.641, 1.3; gt g,  0.214, 0.9; gt g,  0.214; gt g,  0.214, 0.9; gt g,  0.427, 1.1
      gt d,  0.641, 1.3; gt d,  0.214, 0.9; gt d,  0.214; gt d,  0.214, 0.9; gt d,  0.427, 1.1
      gt d,  0.641, 1.3; gt d,  0.214, 0.9; gt d,  0.214; gt d,  0.214, 0.9; gt d,  0.427, 1.1
    end
  end
end


# Ritmo del piano
# Sigue el formato de la funcion p (nota, sleep, release, amp)

# Función del inicio del coro
define :coro_1 do
  p :b4, 0.5; p :b4, 0.5; p :cs5, 0.5, 0.4, 0.7; p :d5, 1; p :d5, 1; p :b4, 1; p :a4, 0.5, 0.5; p :a4, 0.5, 0.5; p :d4, 2, 4, 0.7
end

define :piano_part do
  use_synth :piano
  
  # Parte 1: "Feelin' my way through the darkness"
  p :b4, 0.5, 0.6, 1.0; p :b4, 0.25, 0.6, 1.0; p :b4, 0.75, 0.6, 1.0; p :a4, 0.5; p :fs4, 0.25, 0.4, 0.7; p :a4, 0.75; p :fs4, 0.25, 0.2, 1.0; p :e4, 0.25, 0.2, 1.0
  with_fx :reverb, room: 0.85, mix: 0.5 do; p :d4, 3.5, 3.0, 0.9; end
  
  # Parte 2: "Guided by a beating heart"
  p :b4, 0.5, 0.6, 1.0; p :b4, 0.25, 0.6, 1.0; p :b4, 0.5, 0.6, 1.0; p :a4, 0.5; p :a4, 0.5; p :fs4, 0.5, 0.4, 1.0; p :a4, 0.5
  with_fx :reverb, room: 0.8 do; p :b4, 3.5, 3.5, 0.8; end
  
  # Parte 3: "I can't tell where the journey will end"
  p :b4, 0.25, 0.6, 1.0; p :b4, 0.5, 0.6, 1.0; p :b4, 0.75, 0.6, 1.0; p :a4, 0.5; p :fs4, 0.25, 0.4, 1.0; p :a4, 0.75, 0.6, 0.8; p :fs4, 0.5, 0.5, 0.7; p :e4, 0.25, 0.4, 1.0
  with_fx :reverb, room: 0.85, mix: 0.5 do; p :d4, 3.5, 5, 0.9; end
  
  # Parte 4: "But I know where to start..."
  p :e4, 0.25, 0.4, 1.0; p :fs4, 0.5, 0.4, 1.0; p :g4, 0.75, 0.6, 0.8; p :a4, 0.5; p :fs4, 0.25, 0.4, 1.0; p :e4, 0.5, 0.6, 0.8
  with_fx :reverb, room: 0.85, mix: 0.6 do; p :d4, 3.5, 4, 1.0; end
  
  # Parte 5: "They tell me I'm too young to understand"
  p :b4, 0.25, 0.6, 1.0; p :b4, 0.5, 0.6, 1.0; p :b4, 0.75, 0.6, 1.0; p :a4, 0.5; p :a4, 0.5; p :fs4, 0.25, 0.4, 1.0; p :a4, 0.75, 0.6, 0.8; p :fs4, 0.5, 0.5, 0.7; p :e4, 0.25, 0.4, 1.0
  with_fx :reverb, room: 0.85, mix: 0.5 do; p :d4, 3, 5, 0.9; end
  
  # Parte 6: "They say I'm caught up in a dream"
  p :b4, 0.25, 0.6, 1.0; p :b4, 0.5, 0.6, 1.0; p :b4, 0.75, 0.6, 1.0; p :a4, 0.5; p :a4, 0.5; p :fs4, 0.25, 0.4, 1.0; p :a4, 0.75, 0.6, 0.8
  with_fx :reverb, room: 0.8, mix: 0.5 do; p :b4, 3, 3, 0.85; end
  
  # Parte 7: "Well, life will pass me by if I don't open up my eyes"
  p :b4, 0.5; p :cs5, 0.25, 0.4, 1.0; p :d5, 0.5, 0.6, 0.8; p :cs5, 0.5, 0.4, 1.0; p :b4, 0.5; p :a4, 0.5
  p :a4, 0.5, 0.6, 0.8; p :fs4, 0.5, 0.4, 1.0; p :e4, 0.5; p :d4, 0.5; p :d4, 0.5; p :fs4, 0.5, 0.4, 1.0; p :fs4, 0.5
  
  # Parte 8: "Well, that's fine by me"
  p :e4, 0.25, 0.4, 1.0; p :d4, 0.75, 0.6, 0.8; p :e4, 0.25, 0.4, 1.0; p :fs4, 0.5, 0.4, 1.0; p :g4, 0.75; p :a4, 0.75, 0.6, 0.8; p :e4, 0.5, 0.4, 1.0
  with_fx :reverb, room: 0.8, mix: 0.5 do; p :d4, 2, 3, 0.85; end
  
  # Part 9 (coro): "So wake me up when it's all over"
  coro_1
  
  # Part 10 (coro): "When I'm wiser and I'm older"
  p :b4, 0.5; p :cs5, 0.5, 0.4, 0.7; p :d5, 1; p :d5, 1; p :b4, 1, 0.6, 0.8; p :a4, 0.5, 0.6, 0.8; p :b4, 0.5, 0.6, 1.0; p :a4, 2, 1.8, 0.7
  
  # Part 11 (coro): "All this time I was finding myself, and I"
  p :b4, 0.5; p :cs5, 0.5, 0.4, 0.7; p :d5, 1; p :fs5, 0.5, 0.4, 0.7; p :fs5, 0.5, 0.4, 0.7; p :e5, 0.5, 0.6, 0.8; p :d5, 0.5; p :e5, 0.5, 0.6, 0.8
  p :d4, 1.25, 1, 0.5; p :a4, 0.25, 1.8, 0.7; p :a4, 2, 1.8, 0.7
  
  # Part 12 (coro): "Didn't know I was lost"
  p :a4, 0.75, 0.6, 0.8; p :g4, 0.5, 0.6, 0.8; p :fs4, 0.5, 0.4, 1.0; p :e4, 0.5; p :d4, 2, 1.8, 0.7
  
  # Part 13 (rep. coro): "So wake me up when it's all over"
  coro_1
  
  # Part 14 (rep. coro): "When I'm wiser and I'm older"
  p :b4, 0.5; p :cs5, 0.5, 0.4, 0.7; p :d5, 1; p :fs5, 1, 0.4, 0.7; p :g5, 1; p :a5, 0.5, 0.6, 0.8; p :e5, 0.5, 0.6, 0.8; p :d5, 1.0, 1.8, 0.7; p :e5, 0.25, 1.8, 0.7; p :fs5, 1.5, 1, 0.7
  
  # Part 15 (rep. coro): "All this time I was finding myself, and I"
  p :b4, 0.5; p :cs5, 0.5, 0.4, 0.7; p :d5, 1; p :fs5, 0.5, 0.4, 0.7; p :fs5, 0.5, 0.4, 0.7; p :g5, 0.75; p :g5, 0.5; p :a5, 0.5
  play :e5, release: 0.8, amp: 1.5, cutoff: 40; sleep 0.75
  play :d5, release: 1.8, amp: 1.5, cutoff: 40; sleep 1.5
  
  # Part 16 (rep. coro): "Didn't know I was lost"
  p :a4, 0.25, 0.4, 1.0; p :b4, 1, 0.8, 0.5; p :a4, 1.25, 1.8, 0.5; p :g4, 0.5; p :g4, 0.5; p :fs4, 0.5, 0.4, 0.7; p :fs4, 0.5, 0.4, 0.7; p :a4, 0.5, 0.6, 0.8; p :e4, 0.5, 0.6, 0.8; p :d4, 3, 4, 0.8
end


# Percusion

define :percussion_chorus do
  with_fx :level, amp: 0.3 do
    15.times do
      4.times do
        sample :bd_haus, amp: 0.3, release: 0.25, rate: 0.4
        synth :sine, note: :d1, release: 0.2, amp: 0.1
        sleep 1
      end
    end
  end
end


# Ejecución

# Inicio de la intro en guitarra
guitar_intro

# Entran el piano y la guitarra principal al tiempo

in_thread do
  with_fx :level, amp: 0.3 do
    guitar_main_rhythm
  end
end

in_thread do
  with_fx :level, amp: 2.6 do
    piano_part
  end
end

# Inicia la percusión

in_thread do
  sleep 53   # espera a que el piano llegue al coro
  with_fx :level, amp: 2.0 do
    percussion_chorus
  end
end
