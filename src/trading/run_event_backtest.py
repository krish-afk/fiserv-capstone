import json
import pandas as pd
from pathlib import Path

from src.utils.config import config
from src.trading.strategy import load_configured_strategy
from src.trading.pipeline import build_strategy_data

def evaluate_event_contracts(signals_df: pd.DataFrame, contract_price: float = 0.50):
    """
    Evaluates binary event contracts.
    Resolves trades based on bbg_actual vs bbg_strike stored in the signal metadata.
    """
    results = []
    
    # Read global variables dynamically from the config file.
    # We use .get() with defaults just to be safe if the config structure shifts.
    initial_cash = config.get("trading", {}).get("portfolio", {}).get("initial_cash", 100000.0)
    commission = config.get("trading", {}).get("portfolio", {}).get("commission", 0.002)
    
    capital_pure = initial_cash
    capital_real = initial_cash

    # Risk Management: Allocate exactly 5% of current portfolio capital per trade
    trade_size = capital_pure * 0.05 
    contracts_per_trade = trade_size / contract_price

    for date, row in signals_df.iterrows():
        signal = row["signal"]
        
        # If signal is 0, no trade was triggered, skip to the next day
        if signal == 0:
            continue 

        meta = json.loads(row["metadata"])
        actual = meta["bbg_actual"]
        strike = meta["bbg_strike"]
        
        # Did our contract settle at $1.00 (Win) or $0.00 (Loss)?
        if signal == 1.0: # We bought YES
            won = actual > strike
        else:             # We bought NO
            won = actual <= strike

        # PnL Math per contract
        gross_profit_per_contract = (1.0 - contract_price) if won else -contract_price
        
        trade_pnl_pure = gross_profit_per_contract * contracts_per_trade
        trade_pnl_real = trade_pnl_pure - (commission * contracts_per_trade)

        capital_pure += trade_pnl_pure
        capital_real += trade_pnl_real

        results.append({
            "Date": date.date() if hasattr(date, 'date') else date,
            "Signal": "BUY YES" if signal == 1 else "BUY NO",
            "Model_Pred_Pct": meta["y_pred_pct"],  
            "Model_Pred_Abs": meta["y_pred_abs"],  
            "Strike": strike,
            "Actual": actual,
            "Edge": meta["edge"],
            "Outcome": "WIN" if won else "LOSS",
            "Pure_PnL": round(trade_pnl_pure, 2),
            "Real_PnL": round(trade_pnl_real, 2),
            "Portfolio_Value_Real": round(capital_real, 2)
        })

    results_df = pd.DataFrame(results)
    
    print("\n" + "="*50)
    print("📈 PREDICTION MARKET BACKTEST RESULTS")
    print("="*50)
    if not results_df.empty:
        wins = len(results_df[results_df['Outcome'] == 'WIN'])
        total_trades = len(results_df)
        print(f"Total Trades Taken: {total_trades}")
        print(f"Win Rate:           {(wins / total_trades) * 100:.1f}%")
        print(f"Pure Return:        {((capital_pure - initial_cash) / initial_cash) * 100:.2f}% (0-fee)")
        print(f"Realistic Return:   {((capital_real - initial_cash) / initial_cash) * 100:.2f}% (IBKR fees)")
    else:
        print("No trades triggered. You may need to lower your edge_threshold in config.yaml.")
    print("="*50 + "\n")
    
    return results_df


def run_event_pipeline():
    print("[INFO] Initializing Data for Event Strategy...")
    strategy = load_configured_strategy(cfg=config)
    data = build_strategy_data(strategy=strategy, cfg=config)
    
    print("[INFO] Generating Signals...")
    signals_df = strategy.generate_signals(data)
    
    print("[INFO] Resolving Prediction Market Contracts...")
    # Read contract price from the strategy instance params
    c_price = getattr(strategy, 'contract_price', 0.50)
    results_df = evaluate_event_contracts(signals_df, contract_price=c_price)
    
    out_path = Path("experiments/mrts_trading_results.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not results_df.empty:
        results_df.to_csv(out_path, index=False)
        print(f"[INFO] Trade log saved to {out_path}")
    else:
        print("[INFO] No trades generated, skipping CSV save.")


if __name__ == "__main__":
    run_event_pipeline()