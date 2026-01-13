import joblib
import os

print("Loading model...")
model = joblib.load('model.pkl')
print(f"Original size: {os.path.getsize('model.pkl')/1024/1024:.2f} MB")

print("Compressing...")
joblib.dump(model, 'model.pkl', compress=9, protocol=4)

print("Done.")
print(f"New size: {os.path.getsize('model.pkl')/1024/1024:.2f} MB")
