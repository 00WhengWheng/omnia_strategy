class PortfolioManager:
    """Gestisce l'allocazione del portfolio e il risk management"""
    def __init__(
        self,
        capital: float,
        risk_free_rate: float = 0.02,
        max_position_size: float = 0.2,
        max_correlation: float = 0.7,
        risk_target: float = 0.15
    ):
        self.capital = capital
        self.risk_free_rate = risk_free_rate
        self.max_position_size = max_position_size
        self.max_correlation = max_correlation
        self.risk_target = risk_target
        self.positions = {}
        
    def optimize_portfolio(self, strategies: Dict[str, pd.Series]) -> Dict[str, float]:
        """
        Ottimizza l'allocazione del portfolio usando Black-Litterman
        
        Args:
            strategies: Dict con returns delle strategie
            
        Returns:
            Dict con pesi ottimali
        """
        returns_df = pd.DataFrame(strategies)
        
        # Calcola matrice di covarianza
        cov_matrix = returns_df.cov() * 252
        
        # Calcola rendimenti attesi
        exp_returns = returns_df.mean() * 252
        
        # Implementa Black-Litterman
        weights = self._black_litterman_optimize(exp_returns, cov_matrix)
        
        # Applica vincoli
        weights = self._apply_constraints(weights, cov_matrix)
        
        return dict(zip(strategies.keys(), weights))
    
    def _black_litterman_optimize(
        self, 
        exp_returns: pd.Series, 
        cov_matrix: pd.DataFrame,
        tau: float = 0.05
    ) -> np.ndarray:
        """Implementa il modello Black-Litterman"""
        n_assets = len(exp_returns)
        
        # Prior (market equilibrium)
        prior = np.ones(n_assets) / n_assets
        
        # Views matrix
        P = np.eye(n_assets)
        
        # Views confidence
        omega = np.diag(np.diagonal(tau * cov_matrix))
        
        # Combine prior and views
        pi = exp_returns
        posterior_cov = np.linalg.inv(
            np.linalg.inv(tau * cov_matrix) + 
            P.T @ np.linalg.inv(omega) @ P
        )
        posterior_ret = posterior_cov @ (
            np.linalg.inv(tau * cov_matrix) @ prior + 
            P.T @ np.linalg.inv(omega) @ pi
        )
        
        return self._optimize_weights(posterior_ret, posterior_cov)
    
    def _optimize_weights(
        self, 
        returns: np.ndarray, 
        cov_matrix: np.ndarray
    ) -> np.ndarray:
        """Ottimizza i pesi usando mean-variance optimization"""
        def objective(weights):
            portfolio_ret = np.sum(returns * weights)
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            sharpe = (portfolio_ret - self.risk_free_rate) / portfolio_vol
            return -sharpe
        
        n_assets = len(returns)
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # sum to 1
            {'type': 'ineq', 'fun': lambda x: x}  # non-negative
        ]
        bounds = tuple((0, self.max_position_size) for _ in range(n_assets))
        
        result = minimize(
            objective,
            x0=np.ones(n_assets) / n_assets,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x
    
    def _apply_constraints(
        self, 
        weights: np.ndarray, 
        cov_matrix: pd.DataFrame
    ) -> np.ndarray:
        """Applica vincoli di rischio al portfolio"""
        # Controlla correlazione
        corr_matrix = cov_matrix.corr()
        high_corr = np.where(
            (corr_matrix > self.max_correlation) & 
            (corr_matrix < 1.0)
        )
        
        for i, j in zip(*high_corr):
            weights[j] *= (1 - (corr_matrix.iloc[i, j] - self.max_correlation))
            
        # Normalizza pesi
        weights = weights / np.sum(weights)
        
        # Controlla volatilità target
        port_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        if port_vol > self.risk_target:
            scaling = self.risk_target / port_vol
            weights *= scaling
            
        return weights
    
    def calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calcola metriche di rischio avanzate"""
        metrics = {
            'var_95': self._calculate_var(returns),
            'cvar_95': self._calculate_cvar(returns),
            'omega_ratio': self._calculate_omega_ratio(returns),
            'sortino_ratio': self._calculate_sortino_ratio(returns),
            'calmar_ratio': self._calculate_calmar_ratio(returns),
            'burke_ratio': self._calculate_burke_ratio(returns)
        }
        
        return metrics
    
    def _calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calcola Value at Risk usando simulazione storica"""
        return np.percentile(returns, (1 - confidence) * 100)
    
    def _calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calcola Conditional Value at Risk"""
        var = self._calculate_var(returns, confidence)
        return returns[returns <= var].mean()
    
    def _calculate_omega_ratio(
        self, 
        returns: pd.Series, 
        threshold: float = 0.0
    ) -> float:
        """Calcola Omega Ratio"""
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns < threshold]
        
        if len(losses) == 0 or losses.sum() == 0:
            return np.inf
            
        return gains.sum() / losses.sum()
    
    def _calculate_sortino_ratio(
        self, 
        returns: pd.Series,
        target_return: float = 0.0
    ) -> float:
        """Calcola Sortino Ratio"""
        excess_returns = returns - target_return
        downside_returns = returns[returns < target_return]
        
        if len(downside_returns) == 0:
            return np.inf
            
        downside_std = np.sqrt(np.mean(downside_returns ** 2))
        return np.mean(excess_returns) / downside_std if downside_std != 0 else np.inf
    
    def _calculate_calmar_ratio(
        self, 
        returns: pd.Series,
        window: int = 36
    ) -> float:
        """Calcola Calmar Ratio"""
        total_return = (1 + returns).prod() - 1
        max_dd = self._calculate_max_drawdown(returns)
        
        return total_return / abs(max_dd) if max_dd != 0 else np.inf
    
    def _calculate_burke_ratio(
        self, 
        returns: pd.Series,
        window: int = 36
    ) -> float:
        """Calcola Burke Ratio"""
        excess_returns = returns - self.risk_free_rate / 252
        drawdowns = self._get_drawdowns(returns)
        
        if len(drawdowns) == 0:
            return 0.0
            
        return excess_returns.mean() / np.sqrt(np.sum(drawdowns ** 2))
    
    def _get_drawdowns(self, returns: pd.Series) -> np.ndarray:
        """Calcola la serie di drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative / running_max - 1
        return np.abs(drawdowns[drawdowns < 0])

class RiskManager:
    """Gestisce il rischio in tempo reale durante il trading"""
    def __init__(
        self,
        max_drawdown: float = 0.2,
        var_limit: float = 0.02,
        position_limit: float = 0.25,
        correlation_limit: float = 0.7
    ):
        self.max_drawdown = max_drawdown
        self.var_limit = var_limit
        self.position_limit = position_limit
        self.correlation_limit = correlation_limit
        self.current_positions = {}
        
    def check_trade(
        self, 
        trade_size: float,
        strategy_returns: pd.Series,
        portfolio_value: float,
        current_positions: Dict[str, float]
    ) -> Tuple[bool, str]:
        """Verifica se un trade rispetta i limiti di rischio"""
        # Verifica dimensione posizione
        position_size = trade_size / portfolio_value
        if position_size > self.position_limit:
            return False, "Position size exceeds limit"
            
        # Verifica VaR
        var = self._calculate_var(strategy_returns)
        if abs(var) > self.var_limit:
            return False, "VaR exceeds limit"
            
        # Verifica drawdown
        if self._calculate_drawdown(strategy_returns) > self.max_drawdown:
            return False, "Maximum drawdown exceeded"
            
        # Verifica correlazione
        if not self._check_correlation(strategy_returns, current_positions):
            return False, "Correlation limit exceeded"
            
        return True, "Trade approved"
    
    def _calculate_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
        horizon: int = 1
    ) -> float:
        """Calcola VaR usando simulazione storica"""
        return np.percentile(returns, (1 - confidence) * 100) * np.sqrt(horizon)
    
    def _calculate_drawdown(self, returns: pd.Series) -> float:
        """Calcola il drawdown corrente"""
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative / running_max - 1
        return abs(drawdown.min())
    
    def _check_correlation(
        self,
        new_returns: pd.Series,
        current_positions: Dict[str, pd.Series]
    ) -> bool:
        """Verifica se la nuova strategia è troppo correlata con le posizioni esistenti"""
        if not current_positions:
            return True
            
        for pos_returns in current_positions.values():
            corr = new_returns.corr(pos_returns)
            if abs(corr) > self.correlation_limit:
                return False
                
        return True

# Esempio di utilizzo del sistema completo
def run_trading_system():
    # Carica dati
    data = pd.read_csv('market_data.csv', index_col='Date', parse_dates=True)
    
    # Inizializza sistema
    system = IntegratedTradingSystem(
        data=data,
        price_columns=['Open', 'High', 'Low', 'Close'],
        volume_columns=['Volume']
    )
    
    # Esegui analisi completa
    results = system.run_full_analysis()
    
    # Inizializza Portfolio Manager
    portfolio_manager = PortfolioManager(
        capital=1000000,
        risk_target=0.15
    )
    
    # Ottimizza allocazione
    strategies = {
        'main_strategy': results['strategy_config'],
        # Aggiungi altre strategie se necessario
    }
    
    optimal_weights = portfolio_manager.optimize_portfolio(strategies)
    
    # Inizializza Risk Manager
    risk_manager = RiskManager(
        max_drawdown=0.2,
        var_limit=0.02
    )
    
    # Monitora rischio in tempo reale
    for strategy, weight in optimal_weights.items():
        is_safe, message = risk_manager.check_trade(
            trade_size=weight * portfolio_manager.capital,
            strategy_returns=results['performance_metrics']['returns'],
            portfolio_value=portfolio_manager.capital,
            current_positions=portfolio_manager.positions
        )
        print(f"Strategy {strategy}: {message}")
    
    # Calcola metriche di rischio finali
    risk_metrics = portfolio_manager.calculate_risk_metrics(
        results['performance_metrics']['returns']
    )
    
    return {
        'results': results,
        'optimal_weights': optimal_weights,
        'risk_metrics': risk_metrics
    }

 '''
# Configura e avvia il sistema
    results = run_trading_system()

    # Analizza risultati
    print("Performance Metrics:", results['results']['performance_metrics'])
    print("Optimal Weights:", results['optimal_weights'])
    print("Risk Metrics:", results['risk_metrics'])
'''