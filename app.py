# app.py (Main Streamlit App)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re
from typing import Dict, List, Tuple, Optional

# ==================== MODULE 1: DATA PARSER ====================
class DataParser:
    def __init__(self):
        self.config = {
            'time_strictness': 5,
            'validation_level': 7,
            'allow_partial_blocks': True
        }
    
    def update_config(self, strictness: int, validation: int, allow_partial: bool):
        self.config.update({
            'time_strictness': strictness,
            'validation_level': validation,
            'allow_partial_blocks': allow_partial
        })
    
    def ml_to_probability(self, ml: float) -> float:
        """Convert moneyline to implied probability"""
        if ml > 0:
            return 100 / (ml + 100)
        else:
            return abs(ml) / (abs(ml) + 100)
    
    def parse_timestamp(self, time_str: str) -> datetime:
        """Parse various timestamp formats"""
        if not time_str:
            return datetime.now()
        
        try:
            # Handle common formats
            if 'PM' in time_str.upper() or 'AM' in time_str.upper():
                return datetime.strptime(time_str, '%m/%d %I:%M%p')
            else:
                return datetime.strptime(time_str, '%m/%d %H:%M')
        except:
            return datetime.now()
    
    def parse_odds_data(self, raw_data: str) -> Tuple[pd.DataFrame, List[str]]:
        """Main parsing function - highly tunable"""
        lines = [line.strip() for line in raw_data.split('\n') if line.strip()]
        blocks = []
        errors = []
        
        # Skip header if present
        start_idx = 1 if any(keyword in lines[0].upper() for keyword in ['TIME', 'DATE', 'AWAY', 'HOME']) else 0
        
        for i in range(start_idx, len(lines), 4):
            try:
                block = self._parse_single_block(lines[i:i+4], i)
                if block:
                    blocks.append(block)
            except Exception as e:
                errors.append(f"Line {i}: {str(e)}")
        
        df = pd.DataFrame(blocks)
        if not df.empty:
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df, errors
    
    def _parse_single_block(self, lines: List[str], line_num: int) -> Optional[Dict]:
        """Parse a single 4-line block with configurable strictness"""
        if len(lines) < 4 and not self.config['allow_partial_blocks']:
            return None
        
        # Line 1: Time Team AwayML HomeML Total
        l1 = lines[0].split()
        if len(l1) < 5:
            if self.config['validation_level'] > 8:
                return None
            # Try to handle with fewer elements
            l1 = self._handle_partial_line_1(l1)
        
        time_str = f"{l1[0]} {l1[1]}" if len(l1) >= 2 else None
        away_ml = self._safe_float(l1[2]) if len(l1) >= 3 else None
        home_ml = self._safe_float(l1[3]) if len(l1) >= 4 else None
        total = self._extract_total(l1[4]) if len(l1) >= 5 else None
        
        # Line 2: totalVig
        l2 = lines[1].split() if len(lines) > 1 else []
        total_vig = self._safe_float(l2[0]) if l2 else None
        
        # Line 3: team runline
        l3 = lines[2].split() if len(lines) > 2 else []
        runline = self._safe_float(l3[1]) if len(l3) >= 2 else None
        
        # Line 4: runlineVig
        l4 = lines[3].split() if len(lines) > 3 else []
        runline_vig = self._safe_float(l4[0]) if l4 else None
        
        # Validate based on config
        if not self._validate_block(away_ml, home_ml, total, runline):
            return None
        
        # Convert to probabilities
        away_prob = self.ml_to_probability(away_ml) if away_ml else None
        home_prob = self.ml_to_probability(home_ml) if home_ml else None
        
        return {
            'time': time_str,
            'timestamp': self.parse_timestamp(time_str) if time_str else datetime.now(),
            'away_ml': away_ml,
            'home_ml': home_ml,
            'away_prob': away_prob,
            'home_prob': home_prob,
            'total': total,
            'total_vig': total_vig,
            'runline': runline,
            'runline_vig': runline_vig,
            'raw_data': lines
        }
    
    def _handle_partial_line_1(self, l1: List[str]) -> List[str]:
        """Handle incomplete first lines based on validation level"""
        if self.config['validation_level'] <= 5:
            # Very lenient - try to extract what we can
            while len(l1) < 5:
                l1.append('')
        return l1
    
    def _safe_float(self, value: str) -> Optional[float]:
        """Safely convert to float with error handling"""
        try:
            # Remove any non-numeric characters except decimal and minus
            cleaned = re.sub(r'[^\d.-]', '', value)
            return float(cleaned) if cleaned else None
        except:
            return None
    
    def _extract_total(self, total_str: str) -> Optional[float]:
        """Extract total from o/u format"""
        try:
            # Handle o8.5, u9, O10, etc.
            match = re.search(r'[ou]?([\d.]+)', total_str, re.IGNORECASE)
            return float(match.group(1)) if match else None
        except:
            return None
    
    def _validate_block(self, away_ml: Optional[float], home_ml: Optional[float], 
                       total: Optional[float], runline: Optional[float]) -> bool:
        """Validate block based on configurable strictness"""
        min_required = max(1, self.config['validation_level'] // 2)
        values = [v for v in [away_ml, home_ml, total, runline] if v is not None]
        return len(values) >= min_required


# ==================== MODULE 2: TECHNICAL ANALYSIS ====================
class TechnicalAnalyzer:
    def __init__(self):
        self.config = {
            'ema_fast': 2,
            'ema_slow': 3,
            'rsi_period': 3,
            'atr_period': 3,
            'zscore_window': 5,
            'sensitivity': 6
        }
    
    def update_config(self, **kwargs):
        self.config.update(kwargs)
    
    def calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()
    
    def calculate_rsi(self, series: pd.Series, period: int) -> pd.Series:
        """Relative Strength Index"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_atr(self, series: pd.Series, period: int) -> float:
        """Average True Range (simplified)"""
        if len(series) < period + 1:
            return 0.0
        true_ranges = [abs(series.iloc[i] - series.iloc[i-1]) for i in range(1, len(series))]
        return np.mean(true_ranges[-period:])
    
    def calculate_zscore(self, series: pd.Series, window: int) -> float:
        """Z-score for current value"""
        if len(series) < window:
            return 0.0
        recent = series.tail(window)
        return (series.iloc[-1] - recent.mean()) / recent.std() if recent.std() != 0 else 0.0
    
    def analyze_series(self, series: pd.Series, series_type: str) -> Dict:
        """Comprehensive TA analysis for a series"""
        if len(series) < 2:
            return {}
        
        # Calculate all indicators
        ema_fast = self.calculate_ema(series, self.config['ema_fast'])
        ema_slow = self.calculate_ema(series, self.config['ema_slow'])
        rsi = self.calculate_rsi(series, self.config['rsi_period'])
        atr = self.calculate_atr(series, self.config['atr_period'])
        zscore = self.calculate_zscore(series, self.config['zscore_window'])
        
        # Generate signals based on sensitivity
        signals = self._generate_signals(series, ema_fast, ema_slow, rsi, series_type)
        
        return {
            'current_value': series.iloc[-1],
            'ema_fast': ema_fast.iloc[-1] if not pd.isna(ema_fast.iloc[-1]) else None,
            'ema_slow': ema_slow.iloc[-1] if not pd.isna(ema_slow.iloc[-1]) else None,
            'rsi': rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else None,
            'atr': atr,
            'zscore': zscore,
            'signals': signals,
            'momentum': series.iloc[-1] - series.iloc[-2] if len(series) > 1 else 0
        }
    
    def _generate_signals(self, series: pd.Series, ema_fast: pd.Series, 
                         ema_slow: pd.Series, rsi: pd.Series, series_type: str) -> Dict:
        """Generate trading signals based on current configuration"""
        sensitivity_multiplier = self.config['sensitivity'] / 5.0  # 1-10 scale to multiplier
        
        signals = {}
        
        # EMA crossover signal
        if (not pd.isna(ema_fast.iloc[-1]) and not pd.isna(ema_slow.iloc[-1])):
            ema_signal = 1 if ema_fast.iloc[-1] > ema_slow.iloc[-1] else -1
            signals['ema_crossover'] = ema_signal
        
        # RSI signals with sensitivity adjustment
        if not pd.isna(rsi.iloc[-1]):
            rsi_value = rsi.iloc[-1]
            oversold_threshold = 30 - (sensitivity_multiplier * 5)
            overbought_threshold = 70 + (sensitivity_multiplier * 5)
            
            if rsi_value < oversold_threshold:
                signals['rsi'] = 1  # Oversold - bullish
            elif rsi_value > overbought_threshold:
                signals['rsi'] = -1  # Overbought - bearish
            else:
                signals['rsi'] = 0
        
        # Momentum signal
        if len(series) > 1:
            momentum = series.iloc[-1] - series.iloc[-2]
            momentum_threshold = sensitivity_multiplier * 0.5
            signals['momentum'] = 1 if momentum > momentum_threshold else (-1 if momentum < -momentum_threshold else 0)
        
        return signals


# ==================== MODULE 3: MARKET REGIME DETECTOR ====================
class MarketRegimeDetector:
    def __init__(self):
        self.config = {
            'volatility_threshold_high': 2.0,
            'volatility_threshold_low': 0.5,
            'steam_threshold': 3.0,
            'lookback_window': 5
        }
    
    def update_config(self, volatility_high: float, volatility_low: float, 
                     steam_threshold: float, lookback: int):
        self.config.update({
            'volatility_threshold_high': volatility_high,
            'volatility_threshold_low': volatility_low,
            'steam_threshold': steam_threshold,
            'lookback_window': lookback
        })
    
    def detect_regime(self, df: pd.DataFrame) -> str:
        """Detect current market regime"""
        if len(df) < 3:
            return "INSUFFICIENT_DATA"
        
        # Calculate volatility metrics
        away_probs = df['away_prob'].dropna()
        home_probs = df['home_prob'].dropna()
        
        if len(away_probs) < 2 or len(home_probs) < 2:
            return "INSUFFICIENT_DATA"
        
        volatility = self._calculate_volatility(away_probs, home_probs)
        steam_move = self._detect_steam_move(away_probs, home_probs)
        
        # Determine regime
        if steam_move and volatility > self.config['volatility_threshold_high']:
            return "HIGH_VOLATILITY_STEAM"
        elif volatility > self.config['volatility_threshold_high']:
            return "ELEVATED_VOLATILITY"
        elif volatility < self.config['volatility_threshold_low']:
            return "LOW_VOLATILITY_STABLE"
        else:
            return "NORMAL_VOLATILITY"
    
    def _calculate_volatility(self, away_probs: pd.Series, home_probs: pd.Series) -> float:
        """Calculate market volatility"""
        away_changes = away_probs.diff().abs().mean()
        home_changes = home_probs.diff().abs().mean()
        return (away_changes + home_changes) * 100  # Convert to percentage points
    
    def _detect_steam_move(self, away_probs: pd.Series, home_probs: pd.Series) -> bool:
        """Detect steam moves in probabilities"""
        if len(away_probs) < 2:
            return False
        
        recent_away_move = (away_probs.iloc[-1] - away_probs.iloc[-2]) * 100
        recent_home_move = (home_probs.iloc[-1] - home_probs.iloc[-2]) * 100
        
        return (abs(recent_away_move) > self.config['steam_threshold'] or 
                abs(recent_home_move) > self.config['steam_threshold'])


# ==================== MODULE 4: RECOMMENDATION ENGINE ====================
class RecommendationEngine:
    def __init__(self):
        self.config = {
            'confidence_threshold_high': 0.7,
            'edge_threshold': 0.02,
            'momentum_weight': 0.3,
            'regime_weight': 0.4
        }
    
    def generate_recommendations(self, df: pd.DataFrame, regime: str, 
                               ta_analysis: Dict) -> Dict:
        """Generate context-aware recommendations"""
        if df.empty:
            return {"error": "No data available"}
        
        current = df.iloc[-1]
        initial = df.iloc[0]
        
        # Calculate probability edges
        away_edge = (current['away_prob'] - initial['away_prob']) * 100
        home_edge = (current['home_prob'] - initial['away_prob']) * 100
        
        recommendations = {
            'regime': regime,
            'away_edge': away_edge,
            'home_edge': home_edge,
            'total_prob_shift': abs(away_edge) + abs(home_edge),
            'actions': [],
            'confidence': self._calculate_confidence(df, regime, ta_analysis),
            'market_analysis': self._generate_market_analysis(regime, away_edge, home_edge)
        }
        
        # Generate regime-specific actions
        recommendations['actions'] = self._generate_actions(regime, away_edge, home_edge, ta_analysis)
        
        return recommendations
    
    def _calculate_confidence(self, df: pd.DataFrame, regime: str, ta_analysis: Dict) -> str:
        """Calculate confidence level"""
        data_points = len(df)
        
        if data_points < 3:
            return "LOW"
        elif regime in ["HIGH_VOLATILITY_STEAM", "ELEVATED_VOLATILITY"]:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def _generate_market_analysis(self, regime: str, away_edge: float, home_edge: float) -> str:
        """Generate market analysis text"""
        analyses = {
            "HIGH_VOLATILITY_STEAM": f"🔥 Steam move detected. Away: {away_edge:+.1f}%, Home: {home_edge:+.1f}%",
            "ELEVATED_VOLATILITY": f"📈 Active market with significant probability movement",
            "LOW_VOLATILITY_STABLE": f"⚖️ Stable market with minimal movement",
            "NORMAL_VOLATILITY": f"📊 Normal market activity",
            "INSUFFICIENT_DATA": "📋 Need more data points for analysis"
        }
        return analyses.get(regime, "Analyzing market conditions...")
    
    def _generate_actions(self, regime: str, away_edge: float, home_edge: float, 
                         ta_analysis: Dict) -> List[str]:
        """Generate specific action recommendations"""
        actions = []
        
        if regime == "HIGH_VOLATILITY_STEAM":
            actions.append("Monitor for potential fade opportunities")
            actions.append("Reduce position size due to elevated volatility")
            if away_edge > 2:
                actions.append("Consider Away team if steam confirms trend")
            elif home_edge > 2:
                actions.append("Consider Home team if steam confirms trend")
                
        elif regime == "ELEVATED_VOLATILITY":
            actions.append("Trade with confirmed momentum")
            actions.append("Watch for acceleration patterns")
            
        elif regime == "LOW_VOLATILITY_STABLE":
            actions.append("Look for structural edges in pricing")
            actions.append("Consider smaller, more frequent positions")
            
        else:  # NORMAL_VOLATILITY
            actions.append("Follow trend with proper risk management")
            actions.append("Monitor for regime changes")
        
        return actions


# ==================== MODULE 5: VISUALIZATION ENGINE ====================
class VisualizationEngine:
    def __init__(self):
        self.config = {
            'chart_height': 500,
            'show_technical_indicators': True,
            'color_scheme': 'plotly_dark'
        }
    
    def create_probability_chart(self, df: pd.DataFrame, show_emas: bool = True) -> go.Figure:
        """Create interactive probability evolution chart"""
        fig = go.Figure()
        
        # Add probability lines
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['away_prob'] * 100,
            mode='lines+markers',
            name='Away Probability %',
            line=dict(color='#00ff88', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['home_prob'] * 100,
            mode='lines+markers', 
            name='Home Probability %',
            line=dict(color='#ff4444', width=3)
        ))
        
        # Add EMAs if requested
        if show_emas and len(df) > 3:
            ta = TechnicalAnalyzer()
            away_ema = ta.calculate_ema(df['away_prob'] * 100, 3)
            home_ema = ta.calculate_ema(df['home_prob'] * 100, 3)
            
            fig.add_trace(go.Scatter(
                x=df['time'], y=away_ema,
                mode='lines',
                name='Away EMA(3)',
                line=dict(color='#00cc66', width=1, dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=df['time'], y=home_ema,
                mode='lines',
                name='Home EMA(3)', 
                line=dict(color='#cc3333', width=1, dash='dash')
            ))
        
        fig.update_layout(
            title='Probability Evolution Over Time',
            xaxis_title='Time',
            yaxis_title='Implied Probability %',
            height=400,
            template='plotly_dark'
        )
        
        return fig
    
    def create_analysis_dashboard(self, df: pd.DataFrame, recommendations: Dict, 
                                ta_analysis: Dict) -> None:
        """Create comprehensive analysis dashboard"""
        # This will be implemented in the Streamlit UI
        pass


# ==================== STREAMLIT APP ====================
def main():
    st.set_page_config(
        page_title="Omniscience Pro - Modular Analysis",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize modules
    if 'parser' not in st.session_state:
        st.session_state.parser = DataParser()
        st.session_state.ta = TechnicalAnalyzer()
        st.session_state.regime_detector = MarketRegimeDetector()
        st.session_state.recommendation_engine = RecommendationEngine()
        st.session_state.viz = VisualizationEngine()
    
    st.title("🎯 Omniscience Pro - Modular Sports Analytics")
    
    # Sidebar with module configuration
    with st.sidebar:
        st.header("🔧 Module Configuration")
        
        # Parser Configuration
        st.subheader("📥 Parser Settings")
        time_strictness = st.slider("Time Format Strictness", 1, 10, 5)
        validation_level = st.slider("Data Validation Level", 1, 10, 7)
        allow_partial = st.checkbox("Allow Partial Blocks", True)
        
        # TA Configuration  
        st.subheader("📈 TA Parameters")
        ema_fast = st.slider("EMA Fast Period", 2, 5, 2)
        ema_slow = st.slider("EMA Slow Period", 3, 8, 3)
        sensitivity = st.slider("Market Sensitivity", 1, 10, 6)
        
        # Regime Detection Configuration
        st.subheader("🎯 Regime Detection")
        steam_threshold = st.slider("Steam Threshold", 1.0, 5.0, 2.0)
        volatility_high = st.slider("High Volatility Threshold", 1.0, 5.0, 2.0)
        
        # Update module configurations
        if st.button("Apply Configuration"):
            st.session_state.parser.update_config(time_strictness, validation_level, allow_partial)
            st.session_state.ta.update_config(ema_fast=ema_fast, ema_slow=ema_slow, sensitivity=sensitivity)
            st.session_state.regime_detector.update_config(
                steam_threshold=steam_threshold, 
                volatility_high=volatility_high,
                volatility_low=0.5,
                lookback=5
            )
            st.success("Configuration updated!")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📊 Data Analysis", "🤖 AI Insights", "📈 Charts"])
    
    with tab1:
        st.header("Data Input & Analysis")
        
        # Data input
        raw_data = st.text_area(
            "Paste your odds data (4-line blocks):",
            height=200,
            placeholder="9/5 1:21PM +140 -160 o8.5\n-115\nNYY -1.5\n-110\n..."
        )
        
        if st.button("Analyze Data", type="primary"):
            if raw_data.strip():
                with st.spinner("Analyzing data..."):
                    # Parse data
                    df, errors = st.session_state.parser.parse_odds_data(raw_data)
                    
                    if not df.empty:
                        # Detect market regime
                        regime = st.session_state.regime_detector.detect_regime(df)
                        
                        # Run TA analysis
                        away_probs = df['away_prob'].dropna() * 100
                        home_probs = df['home_prob'].dropna() * 100
                        
                        ta_away = st.session_state.ta.analyze_series(away_probs, 'away_prob')
                        ta_home = st.session_state.ta.analyze_series(home_probs, 'home_prob')
                        
                        # Generate recommendations
                        recommendations = st.session_state.recommendation_engine.generate_recommendations(
                            df, regime, {'away': ta_away, 'home': ta_home}
                        )
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Market Overview")
                            st.metric("Market Regime", recommendations['regime'])
                            st.metric("Data Points", len(df))
                            st.metric("Confidence", recommendations['confidence'])
                            
                            st.subheader("Probability Analysis")
                            st.metric("Away Probability", f"{df.iloc[-1]['away_prob']*100:.1f}%")
                            st.metric("Home Probability", f"{df.iloc[-1]['home_prob']*100:.1f}%")
                            st.metric("Total Shift", f"{recommendations['total_prob_shift']:.1f}%")
                        
                        with col2:
                            st.subheader("Technical Analysis")
                            if ta_away:
                                st.write("**Away Team:**")
                                st.write(f"- RSI: {ta_away.get('rsi', 'N/A')}")
                                st.write(f"- Momentum: {ta_away.get('momentum', 0):.2f}")
                                st.write(f"- ATR: {ta_away.get('atr', 0):.3f}")
                            
                            st.subheader("Recommended Actions")
                            for action in recommendations.get('actions', []):
                                st.write(f"• {action}")
                        
                        # Display data table
                        st.subheader("Parsed Data")
                        display_df = df.copy()
                        display_df['away_prob'] = (display_df['away_prob'] * 100).round(1)
                        display_df['home_prob'] = (display_df['home_prob'] * 100).round(1)
                        st.dataframe(display_df[['time', 'away_ml', 'home_ml', 'away_prob', 'home_prob', 'total', 'runline']])
                        
                        # Show errors if any
                        if errors:
                            st.warning(f"Encountered {len(errors)} parsing errors")
                            with st.expander("Show Errors"):
                                for error in errors:
                                    st.write(error)
                    
                    else:
                        st.error("No valid data could be parsed. Please check your input format.")
            else:
                st.warning("Please paste some odds data to analyze.")
    
    with tab2:
        st.header("Advanced Insights")
        # Additional analysis tabs can be added here
    
    with tab3:
        st.header("Interactive Charts")
        if 'df' in locals() and not df.empty:
            fig = st.session_state.viz.create_probability_chart(df)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
