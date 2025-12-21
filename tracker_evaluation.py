import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import motmetrics as mm
from trackers.pedestrian_tracker import PedestrianTracker


def process_mot_sequence(sequence_path, model_path, tracker_type, output_path):
    seqinfo_path = os.path.join(sequence_path, 'seqinfo.ini')
    if not os.path.exists(seqinfo_path):
        raise FileNotFoundError(f"seqinfo.ini not found at {seqinfo_path}")
    
    original_width = None
    original_height = None
    with open(seqinfo_path, 'r') as f:
        for line in f:
            if line.startswith('imWidth='):
                original_width = int(line.split('=')[1].strip())
            elif line.startswith('imHeight='):
                original_height = int(line.split('=')[1].strip())
    
    if original_width is None or original_height is None:
        raise ValueError("Could not parse imWidth/imHeight from seqinfo.ini")
    
    print(f"Original video resolution: {original_width}x{original_height}")
    
    tracker = PedestrianTracker(
        model_path=model_path,
        tracker_type=tracker_type,
        conf_thresh=0.3,
        iou_thresh=0.45
    )
    
    img_dir = os.path.join(sequence_path, 'img1')
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"img1 directory not found at {img_dir}")
    
    image_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')],
                        key=lambda x: int(x.split('.')[0]))
    
    print(f"Found {len(image_files)} frames")
    
    tracker_width = 1280
    tracker_height = 720
    mot_results = []
    
    for frame_idx, image_file in enumerate(image_files, start=1):
        image_path = os.path.join(img_dir, image_file)
        frame = cv2.imread(image_path)
        
        if frame is None:
            print(f"Could not read frame {frame_idx}: {image_path}")
            continue
        
        frame_height, frame_width = frame.shape[:2]
        scale_x = frame_width / tracker_width
        scale_y = frame_height / tracker_height
        
        processed_frame, tracks = tracker.process_frame(frame)
        
        for track in tracks:
            track_id = track['id']
            bbox = track['bbox']
            confidence = track.get('confidence', 1.0)
            
            x_original = int(round(float(bbox[0]) * scale_x))
            y_original = int(round(float(bbox[1]) * scale_y))
            w_original = int(round(float(bbox[2]) * scale_x))
            h_original = int(round(float(bbox[3]) * scale_y))
            
            x_original = max(0, min(x_original, frame_width - 1))
            y_original = max(0, min(y_original, frame_height - 1))
            w_original = max(1, min(w_original, frame_width - x_original))
            h_original = max(1, min(h_original, frame_height - y_original))
            
            mot_line = f"{frame_idx},{track_id},{x_original},{y_original},{w_original},{h_original},{confidence:.6f},1,-1"
            mot_results.append(mot_line)
        
        if frame_idx % 50 == 0:
            print(f"Processed frame {frame_idx}/{len(image_files)} ({len(tracks)} tracks)")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(mot_results))
    
    print(f"Saved {len(mot_results)} tracking results to {output_path}")
    return len(mot_results)


def evaluate_mot_results(gt_path, result_path, distth=0.5):
    print(f"Loading ground truth from: {gt_path}")
    gt = mm.io.loadtxt(gt_path, fmt='mot15-2D')
    
    print(f"Loading tracking results from: {result_path}")
    res = mm.io.loadtxt(result_path, fmt='mot15-2D')
    
    print(f"GT: {len(gt)} detections across {gt.index.get_level_values('FrameId').nunique()} frames")
    print(f"Results: {len(res)} detections across {res.index.get_level_values('FrameId').nunique()} frames")
    
    print(f"\nComputing metrics with IoU threshold: {distth}")
    acc = mm.utils.compare_to_groundtruth(gt, res, 'iou', distth=distth)
    
    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=['mota', 'motp', 'idf1', 'idp', 'idr', 'num_matches', 'num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations'],
        name='acc'
    )
    
    print("\n" + "="*60)
    print("TRACKING METRICS SUMMARY")
    print("="*60)
    print(f"\nMOTA: {summary['mota'].iloc[0]:.4f}")
    print(f"MOTP: {summary['motp'].iloc[0]:.4f}")
    print(f"IDF1: {summary['idf1'].iloc[0]:.4f}")
    print(f"ID Precision: {summary['idp'].iloc[0]:.4f}")
    print(f"ID Recall: {summary['idr'].iloc[0]:.4f}")
    print(f"\nMatches: {summary['num_matches'].iloc[0]:.0f}")
    print(f"False Positives: {summary['num_false_positives'].iloc[0]:.0f}")
    print(f"Misses: {summary['num_misses'].iloc[0]:.0f}")
    print(f"ID Switches: {summary['num_switches'].iloc[0]:.0f}")
    print(f"Fragmentations: {summary['num_fragmentations'].iloc[0]:.0f}")
    print("="*60 + "\n")
    
    summary_path = result_path.replace('.txt', '_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(str(summary))
    print(f"Summary saved to: {summary_path}")
    
    return summary, acc


def test_mot_tracking(sequence_name, model_path, model_name, tracker_type='both', mot17_base='datasets/mot/MOT17/train'):
    sequence_path = os.path.join(mot17_base, sequence_name)
    gt_path = os.path.join(sequence_path, 'gt', 'gt.txt')
    
    if not os.path.exists(sequence_path):
        print(f"Sequence not found: {sequence_path}")
        return {}
    
    if not os.path.exists(gt_path):
        print(f"Ground truth not found: {gt_path}")
        return {}
    
    trackers_to_test = ['bytetrack', 'deepsort'] if tracker_type == 'both' else [tracker_type]
    results = {}
    
    for tracker in trackers_to_test:
        print(f"\n{'='*60}")
        print(f"Testing {tracker.upper()} on {sequence_name} with {model_name}")
        print(f"{'='*60}")
        
        output_dir = 'output/tracking'
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{model_name}_{tracker}_{sequence_name}.txt")
        
        try:
            process_mot_sequence(sequence_path, model_path, tracker, output_file)
            summary, acc = evaluate_mot_results(gt_path, output_file)
            results[tracker] = {'summary': summary, 'output_file': output_file}
        except Exception as e:
            print(f"Error testing {tracker}: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def compare_trackers(sequence='MOT17-02-DPM', model_name='', tracking_dir='output/tracking', output_dir='output/graphs'):
    def load_tracking_results(txt_path):
        if not os.path.exists(txt_path):
            return None
        return pd.read_csv(txt_path, header=None, names=['frame', 'id', 'x', 'y', 'w', 'h', 'score', 'class', 'visibility'])
    
    def load_metrics(summary_path):
        if not os.path.exists(summary_path):
            return None
        try:
            df = pd.read_csv(summary_path, sep=r'\s+', index_col=0)
            return df.iloc[0]
        except:
            return None
    
    bt_txt = os.path.join(tracking_dir, f'{model_name}_bytetrack_{sequence}.txt')
    ds_txt = os.path.join(tracking_dir, f'{model_name}_deepsort_{sequence}.txt')
    bt_summary = os.path.join(tracking_dir, f'{model_name}_bytetrack_{sequence}_summary.txt')
    ds_summary = os.path.join(tracking_dir, f'{model_name}_deepsort_{sequence}_summary.txt')
    
    if not os.path.exists(bt_txt) or not os.path.exists(ds_txt):
        print(f"Tracking files not available for {model_name}")
        return
    
    bt_df = load_tracking_results(bt_txt)
    ds_df = load_tracking_results(ds_txt)
    
    if bt_df is None or ds_df is None:
        print(f"Could not load tracking data for {model_name}")
        return
    
    bt_metrics = load_metrics(bt_summary)
    ds_metrics = load_metrics(ds_summary)
    
    bt_track_lengths = bt_df.groupby('id').size()
    ds_track_lengths = ds_df.groupby('id').size()
    
    bt_num_tracks = bt_df['id'].nunique()
    ds_num_tracks = ds_df['id'].nunique()
    
    bt_avg_length = bt_track_lengths.mean()
    ds_avg_length = ds_track_lengths.mean()
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.hist(bt_track_lengths, bins=30, alpha=0.6, label='ByteTrack', edgecolor='black', color='#3498db')
    ax1.hist(ds_track_lengths, bins=30, alpha=0.6, label='DeepSORT', edgecolor='black', color='#e74c3c')
    ax1.set_xlabel('Track Length (frames)', fontsize=12)
    ax1.set_ylabel('Number of Tracks', fontsize=12)
    ax1.set_title(f'Track Length Distribution - {model_name}', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    if bt_metrics is not None and ds_metrics is not None:
        metrics = ['mota', 'motp', 'idf1']
        bt_values = [float(bt_metrics[m]) if m in bt_metrics.index else 0 for m in metrics]
        ds_values = [float(ds_metrics[m]) if m in ds_metrics.index else 0 for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        ax2.bar(x - width/2, bt_values, width, label='ByteTrack', color='#3498db', alpha=0.8)
        ax2.bar(x + width/2, ds_values, width, label='DeepSORT', color='#e74c3c', alpha=0.8)
        
        for i, (bt_val, ds_val) in enumerate(zip(bt_values, ds_values)):
            ax2.text(i - width/2, bt_val, f'{bt_val:.3f}', ha='center', va='bottom', fontsize=9)
            ax2.text(i + width/2, ds_val, f'{ds_val:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax2.set_xlabel('Metric', fontsize=12)
        ax2.set_ylabel('Value', fontsize=12)
        ax2.set_title(f'MOTChallenge Metrics - {model_name}', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([m.upper() for m in metrics])
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    graph_filename = f'{model_name}_tracker_comparison.png' if model_name else 'tracker_comparison.png'
    plt.savefig(os.path.join(output_dir, graph_filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*60)
    print(f"TRACKER COMPARISON - {model_name}")
    print("="*60)
    print(f"\nNumber of Tracks:")
    print(f"  ByteTrack: {bt_num_tracks}")
    print(f"  DeepSORT: {ds_num_tracks}")
    
    print(f"\nTrack Length:")
    print(f"  ByteTrack: avg={bt_avg_length:.1f} frames, min={bt_track_lengths.min()}, max={bt_track_lengths.max()}")
    print(f"  DeepSORT: avg={ds_avg_length:.1f} frames, min={ds_track_lengths.min()}, max={ds_track_lengths.max()}")
    
    if bt_metrics is not None and ds_metrics is not None:
        print(f"\nMOT Metrics:")
        if 'mota' in bt_metrics.index:
            print(f"  MOTA - ByteTrack: {float(bt_metrics['mota']):.4f}, DeepSORT: {float(ds_metrics['mota']):.4f}")
        if 'idf1' in bt_metrics.index:
            print(f"  IDF1 - ByteTrack: {float(bt_metrics['idf1']):.4f}, DeepSORT: {float(ds_metrics['idf1']):.4f}")
        if 'num_switches' in bt_metrics.index:
            print(f"  ID Switches - ByteTrack: {int(float(bt_metrics['num_switches']))}, DeepSORT: {int(float(ds_metrics['num_switches']))}")
    
    print("="*60)
    print(f"\nGraph saved to: {output_dir}/{graph_filename}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Tracker Evaluation')
    parser.add_argument('--sequence', type=str, default='MOT17-02-DPM', help='MOT17 sequence name')
    parser.add_argument('--model', type=str, default=None, help='Path to YOLO model')
    parser.add_argument('--tracker', type=str, choices=['bytetrack', 'deepsort', 'both'], default='both', help='Tracker type')
    
    args = parser.parse_args()
    
    if args.model is None:
        models = {
            "best_dataset1": "best_dataset1.pt",
            "best_dataset2": "best_dataset2.pt",
            "best_combined": "best_combined.pt"
        }
        
        print("="*60)
        print("Testing all models on MOT17 sequence")
        print("="*60)
        
        all_results = {}
        for model_name, model_path in models.items():
            if os.path.exists(model_path):
                print(f"\n{'='*60}")
                print(f"Testing {model_name}")
                print(f"{'='*60}")
                
                results = test_mot_tracking(args.sequence, model_path, model_name, args.tracker)
                if results:
                    all_results[model_name] = results
                    compare_trackers(args.sequence, model_name)
            else:
                print(f"Model {model_name} not found at {model_path}, skipping...")
        
        if all_results:
            print(f"\n{'='*60}")
            print("SUMMARY - All Models")
            print(f"{'='*60}")
            for model_name, results in all_results.items():
                print(f"\n{model_name}:")
                for tracker_name, tracker_results in results.items():
                    if 'summary' in tracker_results:
                        summary = tracker_results['summary']
                        print(f"  {tracker_name.upper()}: MOTA={float(summary['mota'].iloc[0]):.4f}, IDF1={float(summary['idf1'].iloc[0]):.4f}")
    else:
        model_name = os.path.basename(args.model).replace('.pt', '')
        test_mot_tracking(args.sequence, args.model, model_name, args.tracker)
        compare_trackers(args.sequence, model_name)
