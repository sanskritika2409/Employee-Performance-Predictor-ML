import pandas as pd
import numpy as np

def generate_data(n=500):
    np.random.seed(42)

    data = pd.DataFrame({
        'age': np.random.randint(22, 60, n),
        'experience': np.random.randint(1, 20, n),
        'department': np.random.choice(['HR', 'IT', 'Sales'], n),
        'salary': np.random.randint(30000, 120000, n),
        'training_hours': np.random.randint(5, 100, n),
        'projects': np.random.randint(1, 10, n)
    })

    data['performance'] = np.where(
        (data['experience'] > 10) & (data['training_hours'] > 50), 'High',
        np.where(data['projects'] > 5, 'Medium', 'Low')
    )

    return data