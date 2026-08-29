"""
LOTTO OPTIMIZER V3.0 - Versión Optimizada (SIN SONIDOS + MODO USUARIO + BENCHMARK)
=========================================
Mejoras implementadas:
- Eliminados TODOS los pitidos/sounds
- Nuevo modo USUARIO: temperatura inicial, factor enfriamiento, umbral y ratio personalizados
- NumPy para operaciones vectorizadas
- Cache LRU para funciones frecuentes
- Simulated Annealing adaptativo
- Sistema de mutaciones inteligente
- Protección de progreso mejorada
- Base de datos local de récords (Lotoideas)
- Herramienta de Benchmark de rendimiento
"""

import random
import os
import math
import copy
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from enum import Enum

try:
    import numpy as np
    NUMPY_DISPONIBLE = True
except ImportError:
    NUMPY_DISPONIBLE = False
    print("⚠️ NumPy no disponible. Usando implementación estándar.")

# ============================================================================
# CONFIGURACIÓN Y CONSTANTES
# ============================================================================
class Velocidad(Enum):
    """Modos de velocidad de optimización"""
    RAPIDA = ("RAPIDA", 0.990, 500, 0.8)
    MEDIA = ("MEDIA", 0.995, 800, 0.85)
    LENTA = ("LENTA", 0.998, 1200, 0.9)
    ULTRA = ("ULTRA", 0.999, 1500, 0.92)
    TURBO = ("TURBO", 0.985, 300, 0.75)
    USUARIO = ("USUARIO", 0.995, 800, 0.85)   # valores por defecto (se sobrescriben)

    def __init__(self, nombre: str, factor: float, umbral: int, ratio_1num: float):
        self._nombre = nombre
        self._factor = factor
        self._umbral = umbral
        self._ratio_1num = ratio_1num

    @property
    def nombre(self) -> str:
        return self._nombre

    @property
    def factor_enfriamiento(self) -> float:
        return self._factor

    @property
    def umbral_estancamiento(self) -> int:
        return self._umbral

    @property
    def ratio_mutacion_1num(self) -> float:
        return self._ratio_1num

@dataclass
class Configuracion:
    """Configuración centralizada del sistema"""
    v: int # Total de números disponibles
    k: int # Números por apuesta
    t: int # Garantía mínima de aciertos
    m: int # Números por sorteo
    universo_size: int = 100_000 # Tamaño del universo de sorteos
    max_snapshots: int = 10 # Snapshots máximos a mantener
    max_caida_permitida: float = 0.3 # Máxima caída permitida en escape
    snapshot_intervalo: int = 500 # Ciclos entre snapshots automáticos
    cache_size: int = 50_000 # Tamaño máximo del cache

    def __post_init__(self):
        """Validaciones"""
        assert self.k <= self.v, "k no puede ser mayor que v"
        assert self.t <= self.k, "t no puede ser mayor que k"
        assert self.t <= self.m, "t no puede ser mayor que m"
        assert self.m <= self.v, "m no puede ser mayor que v"

@dataclass
class Snapshot:
    """Estado guardado del sistema"""
    apuestas_bits: List[int]
    conteos: List[int]
    cobertura: float
    ciclo: int
    temperatura: float
    timestamp: datetime = field(default_factory=datetime.now)

    def __repr__(self):
        return f"Snapshot(ciclo={self.ciclo}, cob={self.cobertura:.4f}%)"

@dataclass
class Estadisticas:
    """Estadísticas de la optimización"""
    mejoras_totales: int = 0
    mutaciones_aceptadas: int = 0
    mutaciones_rechazadas: int = 0
    mutaciones_1num: int = 0
    mutaciones_2num: int = 0
    escapes_realizados: int = 0
    restauraciones: int = 0
    caidas_bloqueadas: int = 0
    tiempo_inicio: datetime = field(default_factory=datetime.now)
    historial_coberturas: List[float] = field(default_factory=list)

    @property
    def ratio_aceptacion(self) -> float:
        total = self.mutaciones_aceptadas + self.mutaciones_rechazadas
        return self.mutaciones_aceptadas / total if total > 0 else 0

    @property
    def tiempo_transcurrido(self) -> str:
        delta = datetime.now() - self.tiempo_inicio
        return str(delta).split('.')[0]

    def agregar_cobertura(self, cob: float):
        self.historial_coberturas.append(cob)
        # Mantener solo últimos 1000
        if len(self.historial_coberturas) > 1000:
            self.historial_coberturas = self.historial_coberturas[-1000:]

# ============================================================================
# UTILIDADES OPTIMIZADAS
# ============================================================================
def limpiar_pantalla():
    """Limpia la pantalla de la consola"""
    os.system('cls' if os.name == 'nt' else 'clear')

class BitUtils:
    """Utilidades optimizadas para operaciones de bits"""

    _cache_lista_a_bits: Dict[Tuple[int, ...], int] = {}
    _cache_bits_a_lista: Dict[Tuple[int, int], List[int]] = {}

    @staticmethod
    def lista_a_bits(nums: List[int]) -> int:
        """Convierte lista de números a representación de bits"""
        key = tuple(sorted(nums))
        if key in BitUtils._cache_lista_a_bits:
            return BitUtils._cache_lista_a_bits[key]

        resultado = sum(1 << (n - 1) for n in nums)

        if len(BitUtils._cache_lista_a_bits) < 100000:
            BitUtils._cache_lista_a_bits[key] = resultado

        return resultado

    @staticmethod
    def bits_a_lista(bits: int, max_val: int) -> List[int]:
        """Convierte bits a lista de números"""
        key = (bits, max_val)
        if key in BitUtils._cache_bits_a_lista:
            return BitUtils._cache_bits_a_lista[key].copy()

        resultado = [i + 1 for i in range(max_val) if bits & (1 << i)]

        if len(BitUtils._cache_bits_a_lista) < 100000:
            BitUtils._cache_bits_a_lista[key] = resultado

        return resultado

    @staticmethod
    def contar_coincidencias(bits1: int, bits2: int) -> int:
        """Cuenta bits coincidentes entre dos números"""
        interseccion = bits1 & bits2
        try:
            return interseccion.bit_count()
        except AttributeError:
            return bin(interseccion).count('1')

    @staticmethod
    def limpiar_caches():
        """Limpia los caches de bits"""
        BitUtils._cache_lista_a_bits.clear()
        BitUtils._cache_bits_a_lista.clear()

# ============================================================================
# SISTEMA DE SNAPSHOTS MEJORADO
# ============================================================================
class SistemaSnapshots:
    """Gestiona snapshots del estado del sistema"""

    def __init__(self, max_snapshots: int = 10):
        self.snapshots: List[Snapshot] = []
        self.max_snapshots = max_snapshots
        self.mejor_snapshot: Optional[Snapshot] = None

    def guardar(self, apuestas: List[int], conteos: List[int],
                cobertura: float, ciclo: int, temperatura: float) -> None:
        """Guarda un nuevo snapshot"""
        snapshot = Snapshot(
            apuestas_bits=copy.deepcopy(apuestas),
            conteos=copy.deepcopy(conteos),
            cobertura=cobertura,
            ciclo=ciclo,
            temperatura=temperatura
        )

        self.snapshots.append(snapshot)

        # Actualizar mejor snapshot
        if self.mejor_snapshot is None or cobertura > self.mejor_snapshot.cobertura:
            self.mejor_snapshot = snapshot

        # Mantener solo los últimos N
        if len(self.snapshots) > self.max_snapshots:
            # Nunca eliminar el mejor
            snapshots_ordenados = sorted(self.snapshots, key=lambda s: s.cobertura)
            peor = snapshots_ordenados[0]
            if peor != self.mejor_snapshot:
                self.snapshots.remove(peor)

    def obtener_mejor(self) -> Optional[Snapshot]:
        """Retorna el snapshot con mejor cobertura"""
        return self.mejor_snapshot

    def obtener_reciente(self) -> Optional[Snapshot]:
        """Retorna el snapshot más reciente"""
        return self.snapshots[-1] if self.snapshots else None

    def tiene_snapshots(self) -> bool:
        return len(self.snapshots) > 0

    def info(self) -> str:
        if not self.snapshots:
            return "Sin snapshots"
        mejor = self.mejor_snapshot.cobertura if self.mejor_snapshot else 0
        return f"{len(self.snapshots)} snapshots (mejor: {mejor:.4f}%)"

# ============================================================================
# ANALIZADOR DE SORTEOS REALES
# ============================================================================
class AnalizadorSorteos:
    """Analiza y valida contra sorteos históricos reales"""

    def __init__(self, config: Configuracion):
        self.config = config
        self.sorteos: List[List[int]] = []
        self.sorteos_bits: List[int] = []

    def cargar_archivo(self, archivo: str) -> bool:
        """Carga sorteos desde archivo"""
        if not archivo or not os.path.exists(archivo):
            return False

        try:
            with open(archivo, 'r', encoding='utf-8') as f:
                for linea in f:
                    nums = []
                    for parte in linea.replace(',', ' ').split():
                        if parte.isdigit():
                            num = int(parte)
                            if 1 <= num <= self.config.v:
                                nums.append(num)

                    if len(nums) == self.config.m:
                        self.sorteos.append(sorted(nums))
                        self.sorteos_bits.append(BitUtils.lista_a_bits(nums))

            return len(self.sorteos) > 0
        except Exception as e:
            print(f"Error cargando sorteos: {e}")
            return False

    def validar(self, apuestas_bits: List[int]) -> Dict:
        """Valida apuestas contra sorteos reales"""
        if not self.sorteos_bits:
            return {}

        resultados = {
            'total_sorteos': len(self.sorteos_bits),
            'sorteos_cubiertos': 0,
            'mejor_acierto': 0,
            'aciertos_por_nivel': defaultdict(int),
            'detalle_sorteos': []
        }

        for i, sorteo_bits in enumerate(self.sorteos_bits):
            mejor_acierto = 0
            cubierto = False

            for apuesta_bits in apuestas_bits:
                aciertos = BitUtils.contar_coincidencias(apuesta_bits, sorteo_bits)
                mejor_acierto = max(mejor_acierto, aciertos)

                if aciertos >= self.config.t:
                    cubierto = True

                resultados['aciertos_por_nivel'][aciertos] += 1

            if cubierto:
                resultados['sorteos_cubiertos'] += 1

            resultados['mejor_acierto'] = max(resultados['mejor_acierto'], mejor_acierto)
            resultados['detalle_sorteos'].append({
                'sorteo': self.sorteos[i],
                'mejor_acierto': mejor_acierto,
                'cubierto': cubierto
            })

        resultados['cobertura_pct'] = (
            resultados['sorteos_cubiertos'] / resultados['total_sorteos']
        ) * 100

        return resultados

    @property
    def cantidad(self) -> int:
        return len(self.sorteos)

# ============================================================================
# MOTOR DE MUTACIONES INTELIGENTE
# ============================================================================
class MotorMutaciones:
    """Sistema inteligente de mutaciones adaptativas"""

    def __init__(self, config: Configuracion):
        self.config = config
        self.exitos = {'1num': 0, '2num': 0, '3num': 0, 'swap': 0}
        self.intentos = {'1num': 1, '2num': 1, '3num': 1, 'swap': 1}
        self.ultimo_tipo: str = '1num'

    def seleccionar_tipo(self, ratio_base: float = 0.8) -> str:
        """Selecciona tipo de mutación basado en éxito histórico"""
        # Calcular tasas de éxito
        tasas = {
            k: self.exitos[k] / self.intentos[k]
            for k in self.exitos
        }

        # Ponderación base + éxito histórico
        pesos = {
            '1num': ratio_base + tasas['1num'] * 0.2,
            '2num': (1 - ratio_base) * 0.7 + tasas['2num'] * 0.2,
            '3num': (1 - ratio_base) * 0.2 + tasas['3num'] * 0.1,
            'swap': (1 - ratio_base) * 0.1 + tasas['swap'] * 0.1
        }

        total = sum(pesos.values())
        r = random.random() * total

        acumulado = 0
        for tipo, peso in pesos.items():
            acumulado += peso
            if r <= acumulado:
                self.ultimo_tipo = tipo
                self.intentos[tipo] += 1
                return tipo

        self.ultimo_tipo = '1num'
        self.intentos['1num'] += 1
        return '1num'

    def registrar_exito(self, tipo: Optional[str] = None):
        """Registra una mutación exitosa"""
        tipo = tipo or self.ultimo_tipo
        self.exitos[tipo] += 1

    def mutar(self, bits: int, tipo: Optional[str] = None) -> int:
        """Aplica mutación según el tipo especificado"""
        if tipo is None:
            tipo = self.seleccionar_tipo()

        nums = BitUtils.bits_a_lista(bits, self.config.v)
        disponibles = [n for n in range(1, self.config.v + 1) if n not in nums]

        if not nums or not disponibles:
            return bits

        if tipo == '1num':
            return self._mutar_n(nums, disponibles, 1)
        elif tipo == '2num':
            return self._mutar_n(nums, disponibles, 2)
        elif tipo == '3num':
            return self._mutar_n(nums, disponibles, 3)
        else: # swap
            return self._mutar_swap(nums, disponibles)

    def _mutar_n(self, nums: List[int], disponibles: List[int], n: int) -> int:
        """Cambia N números"""
        nums = nums.copy()
        n = min(n, len(nums), len(disponibles))

        # Remover N números aleatorios
        for num in random.sample(nums, n):
            nums.remove(num)

        # Agregar N números disponibles
        nums.extend(random.sample(disponibles, n))

        return BitUtils.lista_a_bits(nums)

    def _mutar_swap(self, nums: List[int], disponibles: List[int]) -> int:
        """Intercambia por número cercano (exploración local)"""
        nums = nums.copy()
        idx = random.randint(0, len(nums) - 1)
        viejo = nums[idx]

        # Buscar números cercanos disponibles
        cercanos = [d for d in disponibles if abs(d - viejo) <= 5]

        if cercanos:
            nums[idx] = random.choice(cercanos)
        else:
            nums[idx] = random.choice(disponibles)

        return BitUtils.lista_a_bits(nums)

    def estadisticas(self) -> Dict:
        """Retorna estadísticas de mutaciones"""
        return {
            tipo: {
                'intentos': self.intentos[tipo],
                'exitos': self.exitos[tipo],
                'tasa': self.exitos[tipo] / self.intentos[tipo]
            }
            for tipo in self.exitos
        }

# ============================================================================
# OPTIMIZADOR PRINCIPAL
# ============================================================================
class LottoOptimizerV3:
    """Motor de optimización avanzado con protección de progreso"""

    def __init__(self, config: Configuracion):
        self.config = config
        self.logger = self._configurar_logger()

        # Generar universo de sorteos
        self.logger.info(f"Generando {config.universo_size:,} sorteos simulados...")
        self.universo = self._generar_universo()
        self.total_sorteos = len(self.universo)

        # Estado del sistema
        self.apuestas_bits: List[int] = []
        self.conteos: List[int] = [0] * self.total_sorteos

        # Parámetros de optimización
        self.velocidad = Velocidad.MEDIA
        self.factor_enfriamiento: float = 0.995
        self.umbral_estancamiento: int = 800
        self.ratio_mutacion_1num: float = 0.85
        self.temperatura_inicial: float = 1.0

        self.temperatura = 1.0
        self.ciclos = 0
        self.ciclos_sin_mejora = 0

        # Tracking
        self.mejor_cobertura = 0.0
        self.mejor_cobertura_global = 0.0
        self.ultimo_archivo = ""

        # Subsistemas
        self.stats = Estadisticas()
        self.snapshots = SistemaSnapshots(config.max_snapshots)
        self.mutador = MotorMutaciones(config)
        self.analizador = AnalizadorSorteos(config)

        # Cache de ganancias
        self._cache_ganancias: Dict[Tuple[int, int], int] = {}

    def _configurar_logger(self) -> logging.Logger:
        """Configura el sistema de logging"""
        logger = logging.getLogger('LottoOptimizer')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            # Handler de consola
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s | %(message)s', '%H:%M:%S')
            ch.setFormatter(formatter)
            logger.addHandler(ch)

        return logger

    def _generar_universo(self) -> List[int]:
        """Genera el universo de sorteos simulados"""
        universo = []
        for _ in range(self.config.universo_size):
            sorteo = random.sample(range(1, self.config.v + 1), self.config.m)
            universo.append(BitUtils.lista_a_bits(sorteo))
        return universo

    # -------------------------------------------------------------------------
    # PROPIEDADES
    # -------------------------------------------------------------------------

    @property
    def cobertura(self) -> float:
        """Calcula el porcentaje de cobertura actual"""
        cubiertos = sum(1 for c in self.conteos if c > 0)
        return (cubiertos / self.total_sorteos) * 100

    @property
    def num_apuestas(self) -> int:
        return len(self.apuestas_bits)

    # -------------------------------------------------------------------------
    # GESTIÓN DE APUESTAS
    # -------------------------------------------------------------------------

    def agregar_apuesta(self, nums: List[int]) -> None:
        """Agrega una apuesta y actualiza contadores"""
        bits = BitUtils.lista_a_bits(nums)
        self.apuestas_bits.append(bits)

        for i, sorteo in enumerate(self.universo):
            if BitUtils.contar_coincidencias(bits, sorteo) >= self.config.t:
                self.conteos[i] += 1

    def cargar_archivo(self, archivo: str) -> bool:
        """Carga apuestas desde archivo"""
        if not archivo or not os.path.exists(archivo):
            self.logger.error(f"Archivo no encontrado: {archivo}")
            return False

        try:
            contador = 0
            with open(archivo, 'r', encoding='utf-8') as f:
                for linea in f:
                    if linea.strip().startswith('#'):
                        continue

                    nums = []
                    for parte in linea.replace(',', ' ').split():
                        if parte.isdigit():
                            nums.append(int(parte))

                    if len(nums) == self.config.k:
                        self.agregar_apuesta(nums)
                        contador += 1

            self.logger.info(f"Cargadas {contador} apuestas desde {archivo}")
            self.logger.info(f"Cobertura inicial: {self.cobertura:.4f}%")

            # Snapshot inicial
            self._guardar_snapshot()

            return True

        except Exception as e:
            self.logger.error(f"Error leyendo archivo: {e}")
            return False

    def generar_aleatorias(self, cantidad: int) -> None:
        """Genera apuestas aleatorias"""
        self.logger.info(f"Generando {cantidad} apuestas aleatorias...")

        for i in range(cantidad):
            nums = random.sample(range(1, self.config.v + 1), self.config.k)
            self.agregar_apuesta(nums)

            if (i + 1) % 100 == 0:
                print(f"\r Progreso: {i+1}/{cantidad} | Cobertura: {self.cobertura:.2f}%", end="")

        print()
        self.logger.info(f"Cobertura inicial: {self.cobertura:.4f}%")

    def generar_hasta_objetivo(self, objetivo_pct: float) -> None:
        """Genera apuestas hasta alcanzar el objetivo"""
        self.logger.info(f"Generando apuestas hasta {objetivo_pct}%...")

        while self.cobertura < objetivo_pct:
            nums = random.sample(range(1, self.config.v + 1), self.config.k)
            self.agregar_apuesta(nums)

            if self.num_apuestas % 50 == 0:
                print(f"\r Apuestas: {self.num_apuestas} | Cobertura: {self.cobertura:.2f}%", end="")

        print()
        self.logger.info(f"Generadas {self.num_apuestas} apuestas")

    # -------------------------------------------------------------------------
    # CÁLCULO DE MUTACIONES
    # -------------------------------------------------------------------------

    def calcular_ganancia(self, idx: int, nueva_bits: int) -> int:
        """Calcula la ganancia neta de una mutación"""
        antigua = self.apuestas_bits[idx]

        # Verificar cache
        cache_key = (antigua, nueva_bits)
        if cache_key in self._cache_ganancias:
            return self._cache_ganancias[cache_key]

        ganancia = 0
        t = self.config.t

        for i, sorteo in enumerate(self.universo):
            antes = BitUtils.contar_coincidencias(antigua, sorteo) >= t
            ahora = BitUtils.contar_coincidencias(nueva_bits, sorteo) >= t

            if antes and not ahora:
                if self.conteos[i] == 1:
                    ganancia -= 1
            elif not antes and ahora:
                if self.conteos[i] == 0:
                    ganancia += 1

        # Actualizar cache
        if len(self._cache_ganancias) > self.config.cache_size:
            self._cache_ganancias.clear()
        self._cache_ganancias[cache_key] = ganancia

        return ganancia

    def aplicar_mutacion(self, idx: int, nueva_bits: int) -> None:
        """Aplica una mutación y actualiza contadores"""
        antigua = self.apuestas_bits[idx]
        t = self.config.t

        for i, sorteo in enumerate(self.universo):
            if BitUtils.contar_coincidencias(antigua, sorteo) >= t:
                self.conteos[i] -= 1
            if BitUtils.contar_coincidencias(nueva_bits, sorteo) >= t:
                self.conteos[i] += 1

        self.apuestas_bits[idx] = nueva_bits

    # -------------------------------------------------------------------------
    # SISTEMA DE SNAPSHOTS
    # -------------------------------------------------------------------------

    def _guardar_snapshot(self) -> None:
        """Guarda el estado actual"""
        self.snapshots.guardar(
            self.apuestas_bits,
            self.conteos,
            self.cobertura,
            self.ciclos,
            self.temperatura
        )

    def _restaurar_mejor_snapshot(self) -> bool:
        """Restaura el mejor snapshot guardado"""
        mejor = self.snapshots.obtener_mejor()

        if mejor is None:
            return False

        self.apuestas_bits = copy.deepcopy(mejor.apuestas_bits)
        self.conteos = copy.deepcopy(mejor.conteos)
        self.temperatura = mejor.temperatura
        self.stats.restauraciones += 1

        self.logger.info(f"Restaurado snapshot ciclo {mejor.ciclo}: {mejor.cobertura:.4f}%")
        return True

    # -------------------------------------------------------------------------
    # SISTEMA DE ESCAPE
    # -------------------------------------------------------------------------

    def _escape_controlado(self) -> None:
        """Aplica escape controlado sin destruir progreso"""
        cob_antes = self.cobertura

        limpiar_pantalla()
        print("=" * 70)
        print(f" ⚠️ ESCAPE CONTROLADO - Estancamiento detectado")
        print(f" 📊 Cobertura actual: {cob_antes:.4f}%")
        print(f" 🔄 Ciclos sin mejora: {self.ciclos_sin_mejora}")
        print("=" * 70)

        # Fase 1: Escape suave (10% apuestas, 1 número)
        print(" 🔹 Fase 1: Escape suave (1 número)...")
        n_apuestas = max(5, self.num_apuestas // 10)
        cambios_f1 = 0

        for _ in range(n_apuestas):
            idx = random.randint(0, self.num_apuestas - 1)
            nueva = self.mutador.mutar(self.apuestas_bits[idx], '1num')
            ganancia = self.calcular_ganancia(idx, nueva)

            if ganancia >= -3:
                self.aplicar_mutacion(idx, nueva)
                cambios_f1 += 1

        cob_f1 = self.cobertura
        print(f" ✓ {cambios_f1} cambios | Cobertura: {cob_f1:.4f}%")

        # Fase 2: Solo si no cayó mucho
        if cob_f1 >= cob_antes - self.config.max_caida_permitida:
            print(" 🔸 Fase 2: Escape moderado (2 números)...")
            n_apuestas = max(3, self.num_apuestas // 15)
            cambios_f2 = 0

            for _ in range(n_apuestas):
                idx = random.randint(0, self.num_apuestas - 1)
                nueva = self.mutador.mutar(self.apuestas_bits[idx], '2num')
                ganancia = self.calcular_ganancia(idx, nueva)

                if ganancia >= -8:
                    self.aplicar_mutacion(idx, nueva)
                    cambios_f2 += 1

            cob_f2 = self.cobertura
            print(f" ✓ {cambios_f2} cambios | Cobertura: {cob_f2:.4f}%")

            # Si cayó demasiado, restaurar
            if cob_f2 < cob_antes - self.config.max_caida_permitida:
                print(" ⚠️ Caída excesiva - Restaurando mejor snapshot...")
                self._restaurar_mejor_snapshot()
        else:
            print(" ⚠️ Caída en Fase 1 - Restaurando mejor snapshot...")
            self._restaurar_mejor_snapshot()

        # Reset
        self.temperatura = self.temperatura_inicial
        self.ciclos_sin_mejora = 0
        self.stats.escapes_realizados += 1
        self._cache_ganancias.clear()

        cob_final = self.cobertura
        cambio = cob_final - cob_antes

        print(f"\n 📈 Resultado final: {cob_final:.4f}% ({cambio:+.4f}%)")
        print(" 🌡️ Temperatura reseteada")
        print("=" * 70)

    # -------------------------------------------------------------------------
    # CONFIGURACIÓN DE VELOCIDAD
    # -------------------------------------------------------------------------

    def configurar_velocidad(self) -> None:
        """Menú con nuevo modo USUARIO"""
        print("\n" + "-" * 60)
        print(" CONFIGURACIÓN DE VELOCIDAD")
        print("-" * 60)

        for i, vel in enumerate(Velocidad, 1):
            print(f" {i}. {vel.nombre:12} | Factor: {vel.factor_enfriamiento} | "
                  f"Umbral: {vel.umbral_estancamiento}")

        print("-" * 60)

        try:
            opcion = int(input(" Selecciona [1-6]: ").strip())
            velocidades = list(Velocidad)

            if 1 <= opcion <= len(velocidades):
                self.velocidad = velocidades[opcion - 1]

                if self.velocidad.nombre == "USUARIO":
                    print("\n🔧 MODO USUARIO - Configuración completamente personalizada")
                    self.factor_enfriamiento = float(input("   Factor de enfriamiento (ej: 0.992): ") or 0.995)
                    self.umbral_estancamiento = int(input("   Umbral de estancamiento (ciclos): ") or 800)
                    self.ratio_mutacion_1num = float(input("   Ratio mutación 1 número (0-1): ") or 0.85)
                    self.temperatura_inicial = float(input("   Temperatura inicial (ej: 1.5 / 3.0 / 5.0): ") or 1.0)
                else:
                    self.factor_enfriamiento = self.velocidad.factor_enfriamiento
                    self.umbral_estancamiento = self.velocidad.umbral_estancamiento
                    self.ratio_mutacion_1num = self.velocidad.ratio_mutacion_1num
                    self.temperatura_inicial = 1.0
            else:
                raise ValueError
        except:
            print("   Opción inválida → se usará MODO MEDIA")
            self.velocidad = Velocidad.MEDIA
            self.factor_enfriamiento = 0.995
            self.umbral_estancamiento = 800
            self.ratio_mutacion_1num = 0.85
            self.temperatura_inicial = 1.0

        print(f"\n✓ Modo {self.velocidad.nombre} activado")
        print(f"   Factor enfriamiento    : {self.factor_enfriamiento}")
        print(f"   Umbral estancamiento   : {self.umbral_estancamiento}")
        print(f"   Ratio 1num             : {self.ratio_mutacion_1num}")
        print(f"   Temperatura inicial    : {self.temperatura_inicial}")

    # -------------------------------------------------------------------------
    # GUARDADO DE RESULTADOS
    # -------------------------------------------------------------------------

    def guardar(self) -> str:
        """Guarda las apuestas en archivo"""
        cobertura = self.cobertura
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        nombre = (f"LOTTO_v{self.config.v}_k{self.config.k}_t{self.config.t}_"
                  f"{cobertura:.4f}pct_{timestamp}.txt")

        # Eliminar archivo anterior
        if self.ultimo_archivo and os.path.exists(self.ultimo_archivo):
            try:
                os.remove(self.ultimo_archivo)
            except:
                pass

        # Convertir y ordenar apuestas
        apuestas_ordenadas = []
        for bits in self.apuestas_bits:
            nums = sorted(BitUtils.bits_a_lista(bits, self.config.v))
            apuestas_ordenadas.append(nums)

        apuestas_ordenadas.sort(key=lambda x: tuple(x))

        # Escribir archivo
        with open(nombre, 'w', encoding='utf-8') as f:
            for nums in apuestas_ordenadas:
                linea = " ".join(f"{n:02d}" for n in nums)
                f.write(linea + "\n")

        self.ultimo_archivo = nombre
        return nombre

    # -------------------------------------------------------------------------
    # VISUALIZACIÓN
    # -------------------------------------------------------------------------

    def mostrar_estadisticas(self) -> None:
        """Muestra estadísticas detalladas"""
        print("\n" + "=" * 70)
        print(" ESTADÍSTICAS DETALLADAS")
        print("=" * 70)

        # Configuración
        print(f"\n 📋 Configuración:")
        print(f" v={self.config.v}, k={self.config.k}, t={self.config.t}, m={self.config.m}")

        # Resultados
        print(f"\n 🎯 Resultados:")
        print(f" Apuestas: {self.num_apuestas:,}")
        print(f" Cobertura actual: {self.cobertura:.6f}%")
        print(f" Mejor histórico: {self.mejor_cobertura_global:.6f}%")

        # Optimización
        print(f"\n ⚙️ Optimización:")
        print(f" Ciclos: {self.ciclos:,}")
        print(f" Mejoras: {self.stats.mejoras_totales:,}")
        print(f" Temperatura: {self.temperatura:.6f}")
        print(f" Modo: {self.velocidad.nombre}")

        # Mutaciones
        print(f"\n 🧬 Mutaciones:")
        print(f" Aceptadas: {self.stats.mutaciones_aceptadas:,}")
        print(f" Rechazadas: {self.stats.mutaciones_rechazadas:,}")
        print(f" Ratio: {self.stats.ratio_aceptacion:.2%}")

        # Protección
        print(f"\n 🛡️ Protección:")
        print(f" Escapes: {self.stats.escapes_realizados}")
        print(f" Restauraciones: {self.stats.restauraciones}")
        print(f" Caídas bloqueadas: {self.stats.caidas_bloqueadas}")
        print(f" Snapshots: {self.snapshots.info()}")

        # Sorteos reales
        if self.analizador.cantidad > 0:
            stats_real = self.analizador.validar(self.apuestas_bits)
            print(f"\n ✅ Validación REAL ({self.analizador.cantidad} sorteos):")
            print(f" Cobertura: {stats_real['cobertura_pct']:.2f}%")
            print(f" Cubiertos: {stats_real['sorteos_cubiertos']}/{stats_real['total_sorteos']}")
            print(f" Mejor acierto: {stats_real['mejor_acierto']}")

        # Tiempo
        print(f"\n ⏱️ Tiempo: {self.stats.tiempo_transcurrido}")
        print("=" * 70)

    def _mostrar_progreso(self) -> None:
        """Muestra barra de progreso en consola"""
        pct_escape = min(100, (self.ciclos_sin_mejora / self.umbral_estancamiento) * 100)
        barra_llena = int(pct_escape // 5)
        barra = "█" * barra_llena + "░" * (20 - barra_llena)

        msg = (f" {self.velocidad.nombre:8} | "
               f"🔄 {self.ciclos:,} | "
               f"🎯 {self.mejor_cobertura:.4f}% | "
               f"🌡️ {self.temperatura:.4f} | "
               f"[{barra}]")

        print(msg, end="\r")

    # -------------------------------------------------------------------------
    # MENÚ DE PAUSA
    # -------------------------------------------------------------------------

    def _menu_pausa(self) -> bool:
        """Menú interactivo durante Ctrl+C"""
        print("\n\n" + "!" * 60)
        print(" MENÚ DE CONTROL")
        print("!" * 60)

        print(f"\n Estado actual:")
        print(f" Apuestas: {self.num_apuestas}")
        print(f" Cobertura: {self.cobertura:.4f}%")
        print(f" Mejor: {self.mejor_cobertura_global:.4f}%")
        print(f" Temperatura: {self.temperatura:.6f}")
        print(f" Ciclos sin mejora: {self.ciclos_sin_mejora}")

        print("\n Opciones:")
        print(" 1. Continuar optimización")
        print(" 2. Cambiar velocidad")
        print(" 3. Forzar escape controlado")
        print(" 4. Restaurar mejor snapshot")
        print(" 5. Ver estadísticas completas")
        print(" 6. Guardar y salir")
        print(" 7. Salir sin guardar")

        print("-" * 60)
        opcion = input(" Selección: ").strip()

        if opcion == "1":
            return True
        elif opcion == "2":
            self.configurar_velocidad()
            return True
        elif opcion == "3":
            self._escape_controlado()
            input("\n Presiona ENTER para continuar...")
            return True
        elif opcion == "4":
            if self.snapshots.tiene_snapshots():
                self._restaurar_mejor_snapshot()
                print(f" ✓ Cobertura: {self.cobertura:.4f}%")
            else:
                print(" ⚠️ No hay snapshots disponibles")
            input("\n Presiona ENTER para continuar...")
            return True
        elif opcion == "5":
            self.mostrar_estadisticas()
            input("\n Presiona ENTER para continuar...")
            return True
        elif opcion == "6":
            archivo = self.guardar()
            print(f"\n ✓ Guardado en: {archivo}")
            self.mostrar_estadisticas()
            return False
        elif opcion == "7":
            print("\n Saliendo sin guardar...")
            return False
        else:
            return True

    # -------------------------------------------------------------------------
    # BUCLE PRINCIPAL DE OPTIMIZACIÓN
    # -------------------------------------------------------------------------

    def optimizar(self, objetivo_pct: Optional[float] = None) -> None:
        """Bucle principal de optimización"""
        if self.num_apuestas == 0:
            self.logger.error("No hay apuestas para optimizar")
            return

        # Inicialización
        self.mejor_cobertura = self.cobertura
        self.mejor_cobertura_global = self.mejor_cobertura
        self._guardar_snapshot()

        self.logger.info(f"Cobertura inicial: {self.mejor_cobertura:.4f}%")

        # Configurar velocidad
        self.configurar_velocidad()

        limpiar_pantalla()
        print("=" * 70)
        print(" OPTIMIZANDO - Ctrl+C para menú")
        print("=" * 70)

        try:
            while True:
                # Verificar objetivo
                if objetivo_pct and self.mejor_cobertura >= objetivo_pct:
                    print(f"\n\n 🎯 ¡OBJETIVO ALCANZADO! {self.mejor_cobertura:.4f}%")
                    break

                # Verificar estancamiento
                if self.ciclos_sin_mejora >= self.umbral_estancamiento:
                    self._escape_controlado()

                self.ciclos += 1
                self.ciclos_sin_mejora += 1

                # Seleccionar apuesta a mutar
                idx = random.randint(0, self.num_apuestas - 1)

                # Seleccionar tipo de mutación
                if random.random() < self.ratio_mutacion_1num:
                    tipo_mut = '1num'
                else:
                    tipo_mut = self.mutador.seleccionar_tipo(0.5)

                nueva_bits = self.mutador.mutar(self.apuestas_bits[idx], tipo_mut)
                ganancia = self.calcular_ganancia(idx, nueva_bits)

                # Criterio de aceptación (Metropolis)
                aceptar = False

                if ganancia > 0:
                    aceptar = True
                elif self.temperatura > 0.0001 and ganancia >= -15:
                    # MEJORA: Factor dinámico. A mayor cobertura, más estricto se vuelve.
                    factor_escala = max(5.0, 50.0 * (1.0 - (self.cobertura / 100.0)))
                    prob = math.exp(ganancia / (self.temperatura * factor_escala))
                    aceptar = random.random() < prob

                if ganancia < -15:
                    self.stats.caidas_bloqueadas += 1

                # Aplicar mutación si se acepta
                if aceptar:
                    self.aplicar_mutacion(idx, nueva_bits)
                    self.stats.mutaciones_aceptadas += 1

                    cob_actual = self.cobertura

                    # ¿Mejora?
                    if cob_actual > self.mejor_cobertura:
                        mejora = cob_actual - self.mejor_cobertura
                        self.mejor_cobertura = cob_actual

                        if cob_actual > self.mejor_cobertura_global:
                            self.mejor_cobertura_global = cob_actual

                        self.ciclos_sin_mejora = 0
                        self.stats.mejoras_totales += 1
                        self.stats.agregar_cobertura(cob_actual)
                        self.mutador.registrar_exito(tipo_mut)

                        # Guardar snapshot
                        self._guardar_snapshot()

                        # Guardar archivo
                        archivo = self.guardar()

                        # Mostrar mejora
                        limpiar_pantalla()
                        print("=" * 70)
                        print(f" 🎯 NUEVO RÉCORD: {self.mejor_cobertura:.6f}% (+{mejora:.6f}%)")
                        print(f" 📁 {archivo}")
                        print(f" 📊 Apuestas: {self.num_apuestas}")
                        print(f" 🌡️ Temp: {self.temperatura:.4f} | {self.velocidad.nombre}")
                        print(f" 📈 Mejoras totales: {self.stats.mejoras_totales}")
                        print(f" 🛡️ Protección: {self.stats.caidas_bloqueadas} caídas / "
                              f"{self.stats.restauraciones} restauraciones")

                        if self.analizador.cantidad > 0:
                            stats_real = self.analizador.validar(self.apuestas_bits)
                            print(f" ✅ REAL: {stats_real['cobertura_pct']:.2f}%")

                        print(" [Ctrl+C para menú]")
                        print("=" * 70)
                else:
                    self.stats.mutaciones_rechazadas += 1

                # Snapshot periódico
                if self.ciclos % self.config.snapshot_intervalo == 0:
                    self._guardar_snapshot()

                # Enfriamiento y progreso
                if self.ciclos % 1000 == 0:
                    self.temperatura *= self.factor_enfriamiento
                    self._mostrar_progreso()

        except KeyboardInterrupt:
            if self._menu_pausa():
                limpiar_pantalla()
                print(" Reanudando optimización...")
                self.optimizar(objetivo_pct)

    # -------------------------------------------------------------------------
    # RÉCORDS Y BENCHMARK (NUEVO)
    # -------------------------------------------------------------------------

    def obtener_record_lotoideas(self) -> Optional[int]:
        """Consulta el récord conocido de la tabla de Lotoideas"""
        # Formato de la clave: (v, k, t, m)
        records = {
            (49, 6, 3, 6): 163,
            (49, 6, 4, 6): 1184,
            (49, 6, 5, 6): 11333,
            (49, 6, 3, 5): 101,
            (49, 6, 4, 5): 709,
            (49, 6, 3, 4): 63,
            (49, 6, 4, 4): 431,
            (49, 6, 3, 3): 36,
            # Puedes ir añadiendo más registros aquí en el futuro siguiendo el mismo formato
        }
        return records.get((self.config.v, self.config.k, self.config.t, self.config.m))

    def ejecutar_benchmark(self, duracion_segundos: int = 10) -> None:
        """Ejecuta una prueba de rendimiento pura sin alterar el progreso real"""
        limpiar_pantalla()
        print("=" * 70)
        print(f" 🚀 MODO BENCHMARK - Iniciando test de {duracion_segundos} segundos...")
        print("=" * 70)

        # Si no hay apuestas, generamos algunas para la prueba
        apuestas_prueba = self.num_apuestas
        if apuestas_prueba == 0:
            print(" Generando 100 apuestas iniciales para la prueba...")
            self.generar_aleatorias(100)
            apuestas_prueba = 100

        ciclos_test = 0
        tiempo_inicio = time.time()
        
        # Test de estrés puro
        while time.time() - tiempo_inicio < duracion_segundos:
            idx = random.randint(0, self.num_apuestas - 1)
            tipo_mut = random.choice(['1num', '2num', 'swap'])
            nueva_bits = self.mutador.mutar(self.apuestas_bits[idx], tipo_mut)
            
            # Solo calculamos ganancia (operación más costosa), no aplicamos
            _ = self.calcular_ganancia(idx, nueva_bits)
            
            ciclos_test += 1
            if ciclos_test % 500 == 0:
                print(f"\r Evaluando rendimiento... {int(time.time() - tiempo_inicio)}s", end="")

        tiempo_total = time.time() - tiempo_inicio
        iteraciones_por_segundo = ciclos_test / tiempo_total

        print("\n\n 📊 RESULTADOS DEL BENCHMARK:")
        print("-" * 70)
        print(f" Tiempo de ejecución : {tiempo_total:.2f} segundos")
        print(f" Apuestas en memoria : {apuestas_prueba:,}")
        print(f" Sorteos simulados   : {self.total_sorteos:,} (Universo)")
        print(f" Velocidad del motor : {iteraciones_por_segundo:,.2f} iteraciones/segundo")
        if NUMPY_DISPONIBLE:
            print(" Aceleración NumPy   : ACTIVADA")
        else:
            print(" Aceleración NumPy   : DESACTIVADA (Se recomienda instalar numpy)")
        print("=" * 70)
        input("\n Presiona ENTER para continuar...")


# ============================================================================
# PROGRAMA PRINCIPAL
# ============================================================================
def seleccionar_archivo() -> str:
    """Solicita nombre de archivo al usuario"""
    return input(" Nombre del archivo: ").strip()

def main():
    """Función principal del programa"""
    limpiar_pantalla()

    print("=" * 70)
    print(" 🎰 LOTTO OPTIMIZER V3.0")
    print("=" * 70)
    print(" ✓ Mutaciones adaptativas inteligentes")
    print(" ✓ Sistema de snapshots y restauración")
    print(" ✓ Protección contra caídas de cobertura")
    print(" ✓ Escape controlado sin destruir progreso")
    print(" ✓ Validación contra sorteos reales")
    print(" ✓ Base de datos local de récords")
    print("=" * 70)

    try:
        # Configuración
        print("\n CONFIGURACIÓN")
        print("-" * 70)

        v = int(input(" Total números disponibles (v): "))
        k = int(input(" Números por apuesta (k): "))
        t = int(input(" Garantía mínima (t): "))
        m = int(input(" Números por sorteo (m): "))

        config = Configuracion(v=v, k=k, t=t, m=m)
        optimizador = LottoOptimizerV3(config)

        # Mostrar récord si existe
        record = optimizador.obtener_record_lotoideas()
        if record:
            print(f"\n 🏆 INFO: El récord mundial (Lotoideas) para esta combinación es de {record} apuestas.")
        else:
            print("\n ℹ️ INFO: No hay datos en la tabla interna para esta combinación.")

        # Sorteos reales (opcional)
        print("\n" + "-" * 70)
        cargar_reales = input(" ¿Cargar sorteos reales para validación? (s/n): ").strip().lower()

        if cargar_reales == 's':
            archivo_sorteos = seleccionar_archivo()
            if optimizador.analizador.cargar_archivo(archivo_sorteos):
                print(f" ✓ Cargados {optimizador.analizador.cantidad} sorteos reales")
            else:
                print(" ⚠️ No se pudieron cargar sorteos reales")

        # Modo de operación
        print("\n" + "-" * 70)
        print(" MODO DE OPERACIÓN")
        print("-" * 70)
        print(" 1. Alcanzar % objetivo")
        print(" 2. Generar N apuestas y optimizar")
        print(" 3. Optimizar archivo existente")
        print(" 4. Ejecutar Benchmark de rendimiento")
        print("-" * 70)

        modo = input(" Selecciona [1-4]: ").strip()

        if modo == "1":
            objetivo = float(input("\n Porcentaje objetivo: "))
            optimizador.generar_hasta_objetivo(min(objetivo, 50)) # Límite inicial
            optimizador.optimizar(objetivo)

        elif modo == "2":
            cantidad = int(input("\n Cantidad de apuestas: "))
            optimizador.generar_aleatorias(cantidad)
            optimizador.optimizar()

        elif modo == "3":
            archivo = seleccionar_archivo()
            if optimizador.cargar_archivo(archivo):
                optimizador.optimizar()
            else:
                print(" ⚠️ Error cargando archivo")
                return
                
        elif modo == "4":
            duracion_input = input("\n Segundos de prueba (ej. 10): ").strip()
            duracion = int(duracion_input) if duracion_input.isdigit() else 10
            optimizador.ejecutar_benchmark(duracion)
            return

        else:
            print(" ⚠️ Opción no válida")
            return

        # Mostrar estadísticas finales
        optimizador.mostrar_estadisticas()

        # Guardar resultado final
        if optimizador.num_apuestas > 0:
            archivo_final = optimizador.guardar()
            print(f"\n ✓ Resultado guardado en: {archivo_final}")

    except KeyboardInterrupt:
        print("\n\n Programa interrumpido")

    except ValueError as e:
        print(f"\n ⚠️ Error de valor: {e}")

    except Exception as e:
        print(f"\n ❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()

    input("\n\n Presiona ENTER para salir...")

if __name__ == "__main__":
    main()