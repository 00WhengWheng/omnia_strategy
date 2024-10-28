def _analyze_multiple_timeframes(self, data: pd.DataFrame) -> Dict:
        """Analizza momentum su multipli timeframe"""
        mtf_analysis = {}
        
        for timeframe in self.timeframes:
            # Resample data
            tf_data = self._resample_data(data, timeframe)
            
            # Calcola indicatori per questo timeframe
            indicators = self._calculate_momentum_indicators(tf_data)
            
            # Analizza momentum per questo timeframe
            mtf_analysis[timeframe] = {
                'momentum_score': self._calculate_timeframe_momentum(indicators),
                'trend_alignment': self._check_trend_alignment(indicators),
                'strength': self._calculate_signal_strength(indicators)
            }
            
        return mtf_analysis
        
    def _calculate_timeframe_momentum(self, indicators: Dict) -> float:
        """Calcola score di momentum per singolo timeframe"""
        # RSI momentum
        rsi_score = (indicators['rsi']['value'].iloc[-1] - 50) / 50
        
        # MACD momentum
        macd_score = indicators['macd']['hist'].iloc[-1] / \
                    abs(indicators['macd']['hist']).mean()
        
        # ROC momentum
        roc_score = indicators['roc']['value'].iloc[-1] / self.roc_threshold
        
        # Calcola score composito
        momentum_score = (
            rsi_score * 0.3 +
            macd_score * 0.4 +
            roc_score * 0.3
        )
        
        return np.clip(momentum_score, -1, 1)

    def _check_trend_alignment(self, indicators: Dict) -> float:
        """Verifica allineamento del trend"""
        adx = indicators['adx']['value'].iloc[-1]
        plus_di = indicators['adx']['plus_di'].iloc[-1]
        minus_di = indicators['adx']['minus_di'].iloc[-1]
        
        if adx > self.adx_threshold:
            if plus_di > minus_di:
                return 1.0
            else:
                return -1.0
        return 0.0

    def _calculate_signal_strength(self, indicators: Dict) -> float:
        """Calcola forza del segnale"""
        # ADX strength
        adx_strength = min(indicators['adx']['value'].iloc[-1] / 50, 1.0)
        
        # Volume strength
        volume_strength = min(indicators['mfi']['value'].iloc[-1] / 80, 1.0)
        
        # Momentum consistency
        momentum_cons = self._calculate_momentum_consistency(indicators)
        
        return (adx_strength * 0.4 + 
                volume_strength * 0.3 + 
                momentum_cons * 0.3)

    def _calculate_momentum_consistency(self, indicators: Dict) -> float:
        """Calcola consistenza del momentum"""
        # Verifica consistenza RSI
        rsi_trend = all(indicators['rsi']['value'].tail(5) > 50)
        
        # Verifica consistenza MACD
        macd_trend = all(indicators['macd']['hist'].tail(5) > 0)
        
        # Verifica consistenza ROC
        roc_trend = all(indicators['roc']['value'].tail(5) > 0)
        
        # Calcola score di consistenza
        consistency_score = (
            rsi_trend * 0.3 +
            macd_trend * 0.4 +
            roc_trend * 0.3
        )
        
        return consistency_score

    def _check_momentum_filters(self, data: pd.DataFrame,
                              momentum_signals: Dict,
                              indicators: Dict) -> bool:
        """Applica filtri di momentum"""
        # Verifica forza minima
        if abs(momentum_signals['strength']) < self.min_momentum:
            return False
        
        # Verifica volume
        if self.volume_filter and not self._check_volume_conditions(data):
            return False
        
        # Verifica volatilità
        if self.volatility_filter and not self._check_volatility_conditions(indicators):
            return False
        
        # Verifica correlazione
        if self.correlation_filter and not self._check_correlation_conditions(data):
            return False
        
        return True

    def _calculate_momentum_confidence(self, signals: Dict,
                                    indicators: Dict,
                                    strength: float) -> float:
        """Calcola confidenza nel segnale di momentum"""
        # Trend strength
        trend_confidence = indicators['adx']['trend_strength']
        
        # Signal alignment
        signal_alignment = signals['mtf_consensus']
        
        # Momentum strength
        momentum_conf = strength
        
        # Volume confirmation
        volume_conf = indicators['mfi']['value'].iloc[-1] / 100
        
        # Combine confidence metrics
        confidence = (
            trend_confidence * 0.3 +
            abs(signal_alignment) * 0.3 +
            momentum_conf * 0.2 +
            volume_conf * 0.2
        )
        
        return np.clip(confidence, 0, 1)

    def get_required_columns(self) -> List[str]:
        """Ritorna le colonne richieste dalla strategia"""
        return ['open', 'high', 'low', 'close', 'volume']

    def get_min_required_history(self) -> int:
        """Ritorna il minimo storico richiesto"""
        return max(self.macd_slow + self.macd_signal,
                  self.rsi_period,
                  self.adx_period) + 50

    def _apply_strategy_filters(self, signal: StrategySignal,
                              data: pd.DataFrame) -> bool:
        """Applica filtri specifici della strategia"""
        # Verifica momentum minimo
        if signal.strength < self.min_momentum:
            return False
        
        # Verifica consenso timeframe
        if abs(signal.metadata['mtf_consensus']) < 0.5:
            return False
        
        return True

    def _update_momentum_statistics(self, signals: Dict,
                                  generated_signal: StrategySignal) -> None:
        """Aggiorna statistiche di momentum"""
        new_stats = {
            'timestamp': generated_signal.timestamp,
            'direction': generated_signal.direction,
            'strength': signals['strength'],
            'confidence': generated_signal.confidence,
            'mtf_consensus': signals['mtf_consensus']
        }
        
        self.momentum_history = pd.concat([
            self.momentum_history,
            pd.DataFrame([new_stats])
        ])
