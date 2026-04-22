import preprocess
import enhance
import measure
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_plots(image, idx, curves, best):
    img_label = f"Image {idx+1}"
    labels = ['eme', 'emee', 'ame', 'eme_log', 'visibility', 'amee']
    titles = ['EME', 'EMEE', 'AME', 'EME_LOG', 'Visibility', 'AMEE']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
 
    os.makedirs('plots', exist_ok=True)
    os.makedirs('enhanced', exist_ok=True)
 
    Imin = int(np.min(image))
    Imax = int(np.max(image))
 
    for key, title, color in zip(labels, titles, colors):
        t_opt = best[key][1]
        curve = curves[key]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(256), curve, color=color, linewidth=1.5)
        ax.axvline(t_opt, color='red', linestyle='--', linewidth=1.2, label=f't_opt = {t_opt}')
        ax.scatter([t_opt], [curve[t_opt]], color='red', zorder=5, s=60)
        ax.set_title(f'{img_label} — {title} vs t', fontsize=13, fontweight='bold')
        ax.set_xlabel('Threshold t', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'plots/{img_label}_{title}.png', dpi=130)
        plt.close()

    for key, title in zip(labels, titles):
        t_opt = best[key][1]
        enhanced_img = enhance.contrast_stretching(image, t_opt, Imin, Imax, 0, 255)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(image, cmap='gray', vmin=0, vmax=255)
        axes[0].set_title('Original (preprocessed)', fontsize=11)
        axes[0].axis('off')
        axes[1].imshow(enhanced_img, cmap='gray', vmin=0, vmax=255)
        axes[1].set_title(f'Enhanced — {title} (t_opt={t_opt})', fontsize=11)
        axes[1].axis('off')
        fig.suptitle(f'{img_label} — Optimal Enhanced Image [{title}]', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'enhanced/{img_label}_{title}.png', dpi=130)
        plt.close()

def main():
    images = preprocess.load_dataset("output")
    k1, k2 = 20, 20
    c = 1
    alpha = 1
    Lmin, Lmax = 0, 255

    for idx, image in enumerate(images):
        Imin = np.min(image)
        Imax = np.max(image)

        best = {
            'eme': (-np.inf, 0),
            'emee': (-np.inf, 0),
            'ame': (-np.inf, 0),
            'eme_log': (-np.inf, 0),
            'visibility': (-np.inf, 0),
            'amee': (-np.inf, 0),
        }

        curves = {
            'eme':        np.zeros(256),
            'emee':       np.zeros(256),
            'ame':        np.zeros(256),
            'eme_log':    np.zeros(256),
            'visibility': np.zeros(256),
            'amee':       np.zeros(256),
        }

        # Before enhancement values
        eme_val, emee_val, ame_val, eme_log_val, visibility_val, amee_val = measure.measure_all(image, k1, k2, c, alpha)
        print(f"Image {idx+1} before enhancement metrics:")
        print(f"eme: {eme_val:.4f}")
        print(f"emee: {emee_val:.4f}")
        print(f"ame: {ame_val:.4f}")
        print(f"eme_log: {eme_log_val:.4f}")
        print(f"visibility: {visibility_val:.4f}")
        print(f"amee: {amee_val:.4f}")
        print('\n')

        for t in range(0, 256):
            stretched = enhance.contrast_stretching(image, t, Imin, Imax, Lmin, Lmax)

            eme_val, emee_val, ame_val, eme_log_val, visibility_val, amee_val = measure.measure_all(stretched, k1, k2, c, alpha)

            if eme_val > best['eme'][0]:
                best['eme'] = (eme_val, t)
            if emee_val > best['emee'][0]:
                best['emee'] = (emee_val, t)
            if ame_val > best['ame'][0]:
                best['ame'] = (ame_val, t)
            if eme_log_val > best['eme_log'][0]:
                best['eme_log'] = (eme_log_val, t)
            if visibility_val > best['visibility'][0]:
                best['visibility'] = (visibility_val, t)
            if amee_val > best['amee'][0]:
                best['amee'] = (amee_val, t)

            # Update metric values in curves
            curves['eme'][t]        = eme_val
            curves['emee'][t]       = emee_val
            curves['ame'][t]        = ame_val
            curves['eme_log'][t]    = eme_log_val
            curves['visibility'][t] = visibility_val
            curves['amee'][t]       = amee_val

        print(f"Image {idx+1} best t values:")
        for metric, (val, t_opt) in best.items():
            print(f"  {metric}: t = {t_opt} (value = {val:.4f})")
        print('\n')

        # Generate plots and enhanced images for this image
        generate_plots(image, idx, curves, best)

if __name__ == "__main__":
    main()