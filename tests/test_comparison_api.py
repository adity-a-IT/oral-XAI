import requests

for preset in ['leukoplakia', 'carcinoma', 'variation', 'healthy']:
    try:
        r = requests.post('http://localhost:5000/api/analyze', data={
            'preset': preset,
            'colormap': 'jet',
            'opacity': 0.55,
            'lime': 'true',
            'lime_samples': 20
        }, timeout=30)
        data = r.json()
        print(f"Preset: {preset.upper()} -> Status: {data.get('status')}")
        print(f"  Class: {data.get('predicted_class')} (Index {data.get('predicted_class_index')})")
        print(f"  Confidence: {data.get('confidence')}% | Risk: {data.get('risk_tier')}")
        overlays = data.get('overlays', {})
        for k in ['original_data_url', 'gradcam_data_url', 'gradcampp_data_url', 'lime_data_url']:
            v = overlays.get(k)
            print(f"  Overlay '{k}': {'Present (Base64 length: ' + str(len(v)) + ')' if v else 'MISSING'}")
        print(f"  Telemetry: {data.get('telemetry')}")
        print("-" * 50)
    except Exception as e:
        print(f"Error testing {preset}: {e}")
