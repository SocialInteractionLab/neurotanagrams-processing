import pandas as pd
from datetime import datetime, timedelta

# read csv
df = pd.read_csv('combined.csv')

# Convert 'at' and 'finishedAt' columns to datetime for calculation
df['at'] = pd.to_datetime(df['at'])
df['finishedAt'] = pd.to_datetime(df['finishedAt'])

# check TR intervals and runtime
def check_trs_and_runtime(df):
    results = []
    
    # Group by 'gameId', 'target', 'trialNum' to process each round
    grouped = df.groupby(['gameId', 'target', 'trialNum'])
    
    for name, group in grouped:
        round_start = group[group['verb'] == 'RoundStarted']
        pulses = group[group['verb'] == 'pulse']
        
        # Check if there are enough scanner events (TRs)
        if pulses.empty or round_start.empty:
            results.append((name, 'Missing RoundStarted or pulse events'))
            continue
        
        # Ensure TRs occur every 1.5 seconds
        tr_intervals = pulses['at'].diff().dropna()
        if not all(tr_intervals == timedelta(seconds=1.5)):
            results.append((name, 'Incorrect TR interval'))
            continue
        
        # Calculate runtime from TR timestamps and ensure it matches event output time
        start_time = round_start['at'].iloc[0]
        end_time = group['finishedAt'].iloc[0]
        tr_duration = (pulses['at'].iloc[-1] - pulses['at'].iloc[0]).total_seconds() + 1.5
        runtime_duration = (end_time - start_time).total_seconds()
        
        if not (tr_duration == runtime_duration):
            results.append((name, 'Mismatched runtime duration'))
        else:
            results.append((name, 'Valid'))
    
    return results

results = check_trs_and_runtime(df)

for result in results:
    print(f"Round {result[0]}: {result[1]}")
