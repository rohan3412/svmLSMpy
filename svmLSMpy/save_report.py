import nibabel as nib
import numpy as np
from nilearn.plotting import view_img
from pathlib import Path

from .plot_mosaics import get_slice_coordinates, save_slice_mosaic
from .util import encode_image

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files
  
bg_img_path = files("svmLSMpy.resources").joinpath("mni152.nii.gz")
bg_img = str(bg_img_path)  # Nilearn expects a string path

def save_report(output_file,
                svr_params,
                behaviour_name,
                n_permutations,
                alpha,
                zmap,
                min_patient_count,
                num_patients,
                num_slices,
                nifti_zmap,
                time_taken,
                num_lesions,
                mean_lesion_volume,
                covariates,
                regress_out_lesion_volume,
                regress_out_covariates_on_scores,
                regress_out_covariates_on_lesions,
                normalize_vector,
                n_clusters=5,
                model_name="Support Vector Regression (SVR)",
                suffix=""):
    """
    Save a comprehensive report of the LSM analysis, including parameters, significant voxels, and visualization.
    """
    print("Saving report...")

    # Compute the zmap range
    zmap_min, zmap_max = (np.min(zmap), np.max(zmap))

    max_zscore = max(abs(zmap_min),abs(zmap_max))

    #significant_voxels = np.abs(zmap) > np.percentile(np.abs(zmap), 100 * (1 - alpha))

    output_folder = Path(Path(output_file).parent)

    mosaic_output_folder = output_folder / "mosaics"
    Path(mosaic_output_folder).mkdir(parents=True, exist_ok=True)

    zmap_threshold_output_folder = output_folder / f"thresholded_zmaps{suffix}"

    html_view = view_img(output_folder/f"zmap{suffix}.nii.gz", bg_img=bg_img, threshold=1.7, black_bg=False, cmap="jet")

    # Define file paths (lesion overlap maps are shared across one-vs-rest classes)
    lesion_overlap_path = output_folder / "lesion_overlap.nii.gz"
    lesion_overlap_filtered_path = output_folder / "lesion_overlap_filtered.nii.gz"
    correlation_matrix_path = output_folder / "behavior_covariate_correlation_matrix.png"
    svr_beta_map_path = output_folder / f"beta_map{suffix}.nii.gz"
    zmap_path = output_folder / f"zmap{suffix}.nii.gz"
    zmap_p05_path = zmap_threshold_output_folder / "zmap_p05.nii.gz"
    zmap_p01_path = zmap_threshold_output_folder / "zmap_p01.nii.gz"
    zmap_p005_path = zmap_threshold_output_folder / "zmap_p005.nii.gz"
    zmap_p001_path = zmap_threshold_output_folder / "zmap_p001.nii.gz"
    
    # Define processing configurations
    orientations = ['axial', 'coronal', 'sagittal']
    
    # Maps to store generated mosaics
    mosaics = {}
    
    # Process each orientation
    for orientation in orientations:
        # Get cut coordinates for lesion overlap (used for most maps)
        lesion_cut_coords = get_slice_coordinates(lesion_overlap_path, num_slices, orientation)
        
        # Get cut coordinates for SVR beta map and z-maps
        svr_cut_coords = get_slice_coordinates(svr_beta_map_path, num_slices, orientation)
        
        # Process lesion overlap maps
        output_path = mosaic_output_folder / f"{orientation}_lesion_overlap_mosaic.png"
        save_slice_mosaic(lesion_overlap_path, lesion_cut_coords, output_path, orientation)
        mosaics[f"{orientation}_lesion_overlap"] = encode_image(output_path)
        
        output_path = mosaic_output_folder / f"{orientation}_lesion_overlap_filtered_mosaic.png"
        save_slice_mosaic(lesion_overlap_filtered_path, lesion_cut_coords, output_path, orientation)
        mosaics[f"{orientation}_lesion_overlap_filtered"] = encode_image(output_path)
        
        # Process SVR beta map
        output_path = mosaic_output_folder / f"{orientation}_svr_beta_map_mosaic.png"
        save_slice_mosaic(svr_beta_map_path, svr_cut_coords, output_path, orientation)
        mosaics[f"{orientation}_svr_beta_map"] = encode_image(output_path)
        
        # Process z-maps (with z-score and colormap)
        for zmap_name, zmap_file in [
            ('zmap', zmap_path),
            ('zmap_p05', zmap_p05_path),
            ('zmap_p01', zmap_p01_path),
            ('zmap_p005', zmap_p005_path),
            ('zmap_p001', zmap_p001_path)
        ]:
            output_path = mosaic_output_folder / f"{orientation}_{zmap_name}_mosaic.png"
            save_slice_mosaic(zmap_file, svr_cut_coords, output_path, orientation, max_zscore, cmap='bwr')
            mosaics[f"{orientation}_{zmap_name}"] = encode_image(output_path)
    
    # Extract mosaics to original variable names for backward compatibility
    # AXIAL
    axial_lesion_overlap_mosaic = mosaics['axial_lesion_overlap']
    axial_lesion_overlap_filtered_mosaic = mosaics['axial_lesion_overlap_filtered']
    axial_svr_beta_map_mosaic = mosaics['axial_svr_beta_map']
    axial_zmap_mosaic = mosaics['axial_zmap']
    axial_zmap_p05_mosaic = mosaics['axial_zmap_p05']
    axial_zmap_p01_mosaic = mosaics['axial_zmap_p01']
    axial_zmap_p005_mosaic = mosaics['axial_zmap_p005']
    axial_zmap_p001_mosaic = mosaics['axial_zmap_p001']
    
    # CORONAL
    coronal_lesion_overlap_mosaic = mosaics['coronal_lesion_overlap']
    coronal_lesion_overlap_filtered_mosaic = mosaics['coronal_lesion_overlap_filtered']
    coronal_svr_beta_map_mosaic = mosaics['coronal_svr_beta_map']
    coronal_zmap_mosaic = mosaics['coronal_zmap']
    coronal_zmap_p05_mosaic = mosaics['coronal_zmap_p05']
    coronal_zmap_p01_mosaic = mosaics['coronal_zmap_p01']
    coronal_zmap_p005_mosaic = mosaics['coronal_zmap_p005']
    coronal_zmap_p001_mosaic = mosaics['coronal_zmap_p001']
    
    # SAGITTAL
    sagittal_lesion_overlap_mosaic = mosaics['sagittal_lesion_overlap']
    sagittal_lesion_overlap_filtered_mosaic = mosaics['sagittal_lesion_overlap_filtered']
    sagittal_svr_beta_map_mosaic = mosaics['sagittal_svr_beta_map']
    sagittal_zmap_mosaic = mosaics['sagittal_zmap']
    sagittal_zmap_p05_mosaic = mosaics['sagittal_zmap_p05']
    sagittal_zmap_p01_mosaic = mosaics['sagittal_zmap_p01']
    sagittal_zmap_p005_mosaic = mosaics['sagittal_zmap_p005']
    sagittal_zmap_p001_mosaic = mosaics['sagittal_zmap_p001']


    with open(output_file, 'w') as f:
        f.write(f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Lesion-Symptom Mapping Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #f4f7fc;
                color: #333;
                margin: 20px;
                line-height: 1.7;
            }}
            h1 {{
                color: #004d99;
                font-size: 2.5em;
                margin-bottom: 20px;
            }}
            h2 {{
                color: #004d99;
                font-size: 1.8em;
                border-bottom: 3px solid #004d99;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }}
            h3 {{
                font-size: 1.5em;
                margin-top: 20px;
                color: #2c3e50;
            }}
            p {{
                font-size: 1.1em;
                margin: 15px 0;
                color: #34495e;
            }}
            ul {{
                padding-left: 20px;
                font-size: 1.1em;
            }}
            li {{
                margin-bottom: 10px;
            }}
            .container {{
                width: 85%;
                margin: auto;
                background-color: #fff;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }}
            .section {{
                margin-bottom: 30px;
            }}
            .highlight {{
                background-color: #eaf2f8;
                padding: 15px;
                border-left: 5px solid #1e88e5;
            }}
            .result-table {{
                width: 100%;
                margin-top: 20px;
                border-collapse: collapse;
            }}
            .result-table th, .result-table td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            .result-table th {{
                background-color: #f4f7fc;
                font-weight: bold;
            }}
            .result-table td {{
                background-color: #fafafa;
            }}
            a {{
                color: #004d99;
                text-decoration: none;
            }}
            a:hover {{
                text-decoration: underline;
            }}
            .footer {{
                text-align: center;
                font-size: 0.9em;
                color: #777;
                margin-top: 40px;
            }}
                .image-category img {{
                display: none;
                width: 100%;
                height: auto;
            }}
                .image-category img.active {{
                display: block;
            }}
            .radio-buttons {{
                display: flex;
                gap: 15px;
                margin-top: 20px;
            }}
    
            label {{
                font-size: 16px;
                color: #333;
                cursor: pointer;
                display: flex;
                justify-content: center;
                align-items: center;
                width: 120px;
                height: 40px;
                background-color: #f0f0f0; /* Default background is white */
                transition: all 0.3s ease;
                text-align: center;
                font-weight: bold;
                user-select: none; /* Prevent text selection */
            }}
    
            /* Hide the default radio buttons */
            input[type="radio"] {{
                display: none;
            }}
    
            /* When the radio button is checked, change background to light gray */
            input[type="radio"]:checked + .radio-button-block {{
                background-color: #c9c9c9; /* Light gray background when clicked */
                color: #333;
            }}
    
            /* Style for the block */
            .radio-button-block {{
                width: 100%;
                height: 100%;
                display: flex;
                justify-content: center;
                align-items: center;
                transition: all 0.3s ease;
            }}
    
            /* Hover effect */
            label:hover {{
                background-color: #dfdfdf; /* Light gray hover effect */
            }}
    

        </style>
        <script>
            function switchSvrView(viewPrefix) {{
                const groups = ['group3', 'group4', 'group5', 'group6', 'group7', 'group8'];
                groups.forEach(group => {{
                    const images = document.querySelectorAll(`.image-category img[data-group='${{group}}']`);
                    images.forEach(img => img.classList.remove('active'));
                    const selectedImage = document.querySelector(`.image-category img[data-group='${{group}}'][data-view='${{viewPrefix}}']`);
                    if (selectedImage) {{
                        selectedImage.classList.add('active');
                    }}
                }});
            }}
    
            function switchLesionView(viewPrefix) {{
                const groups = ['group1', 'group2'];
                groups.forEach(group => {{
                    const images = document.querySelectorAll(`.image-category img[data-group='${{group}}']`);
                    images.forEach(img => img.classList.remove('active'));
                    const selectedImage = document.querySelector(`.image-category img[data-group='${{group}}'][data-view='${{viewPrefix}}']`);
                    if (selectedImage) {{
                        selectedImage.classList.add('active');
                    }}
                }});
            }}
    
            // Show or hide the threshold groups based on selected threshold
            function toggleThresholdGroup(threshold) {{
                const thresholdGroups = document.querySelectorAll(`.threshold-group`);
                thresholdGroups.forEach(group => {{
                    if (group.dataset.threshold === threshold) {{
                        group.style.display = 'block';
                    }} else {{
                        group.style.display = 'none';
                    }}
                }});
            }}
    
            window.onload = function() {{
                document.querySelector('input[name="lesionSwitcher"][value="axial"]').click();
                document.querySelector('input[name="svrSwitcher"][value="axial"]').click();
                document.querySelector('input[name="thresholdSwitcher"][value="unthresholded"]').click();
            }};
        </script>
    </head>
    <body>
        <div class="container">
            <h1>svmLSMpy Report for {behaviour_name}</h1>

            <div class="section">
                <h2>Methodology</h2>
                <p>This report analyzes the relationship between lesion status and the behavioral score '{behaviour_name}' using support vector regression for lesion-symptom mapping. {num_patients} binary lesion files in MNI space were analyzed, including only voxels with at least {min_patient_count} overlapping lesions. 
                    {"Lesion volume was controlled using vector normalization." if normalize_vector else "Vector normalization of lesion data was not applied."}
                    
                    {(
                        (
                            "Covariates were regressed out of the behavioral scores. "
                            if regress_out_covariates_on_scores else 
                            "Covariates were not regressed out of the behavioral scores. "
                        )
                        + (
                            "Covariates were regressed out of the lesion data. "
                            if regress_out_covariates_on_lesions else 
                            "Covariates were not regressed out of the lesion data. "
                        )
                        + (
                            "Lesion volume was included as a covariate."
                            if regress_out_lesion_volume else 
                            "Lesion volume was not included as a covariate."
                        )
                    ) if covariates is not None else (
                        (
                            "Regression on behavioral scores was attempted, but no covariates were present. "
                            if regress_out_covariates_on_scores else 
                            "Covariates were not regressed out of the behavioral scores (no covariates present). "
                        )
                        + (
                            "Regression on lesion data was attempted, but no covariates were present."
                            if regress_out_covariates_on_lesions else 
                            "Covariates were not regressed out of the lesion data (no covariates present)."
                        )
                        + (
                            " Lesion volume was included as a covariate."
                            if regress_out_lesion_volume else 
                            " Lesion volume was not included as a covariate."
                        )
                    )}
                    
                    {model_name} was applied with parameters {", ".join(f"{k} = {v}" for k, v in svr_params.items())}, employing grid search optimization.
                    Z maps were derived from null distributions based on {n_permutations} permutations. 
                    
                    The analysis was completed in {time_taken}.
                    </p>
            </div>

            <div class="section highlight">
                <h3>Key Parameters</h3>
                <ul>
                    <li><strong>Number of patients:</strong> {num_patients}</li>
                    <li><strong>Number of permutations:</strong> {n_permutations}</li>
                    <li><strong>Alpha level:</strong> {alpha}</li>
                </ul>
            </div>

            
            <div class="section">
                <h2>Lesion Overlap</h2>
                    <!-- First Set of Radio Buttons (Lesion Switcher) -->
                    <div class="radio-buttons">
                        <label>
                            <input type="radio" name="lesionSwitcher" value="axial" onclick="switchLesionView('axial');" checked>
                            <div class="radio-button-block">Axial</div>
                        </label>
                        <label>
                            <input type="radio" name="lesionSwitcher" value="coronal" onclick="switchLesionView('coronal');">
                            <div class="radio-button-block">Coronal</div>
                        </label>
                        <label>
                            <input type="radio" name="lesionSwitcher" value="sagittal" onclick="switchLesionView('sagittal');">
                            <div class="radio-button-block">Sagittal</div>
                        </label>
                    </div>
                    <!-- Lesion Overlap Group (1st div) -->
                    <div class="image-category">
                        <h3>Unfiltered</h3>
                        <img src="data:image/png;base64,{axial_lesion_overlap_mosaic}" alt="Lesion Overlap Axial" data-group="group1" data-view="axial" style="width: 100%; height: auto;" class="active">
                        <img src="data:image/png;base64,{coronal_lesion_overlap_mosaic}" alt="Lesion Overlap Coronal" data-group="group1" data-view="coronal" style="width: 100%; height: auto;">
                        <img src="data:image/png;base64,{sagittal_lesion_overlap_mosaic}" alt="Lesion Overlap Sagittal" data-group="group1" data-view="sagittal" style="width: 100%; height: auto;">
                    </div>
                """)
        if min_patient_count>0:
            f.write(f"""
                    <br><br>
                
                    <!-- Lesion Overlap Filtered Group (2nd div) -->
                    <div class="image-category">
                        <h3>Filtered (Minimum {min_patient_count} patients)</h3>
                        <img src="data:image/png;base64,{axial_lesion_overlap_filtered_mosaic}" alt="Lesion Overlap Filtered Axial" data-group="group2" data-view="axial" style="width: 100%; height: auto;" class="active">
                        <img src="data:image/png;base64,{coronal_lesion_overlap_filtered_mosaic}" alt="Lesion Overlap Filtered Coronal" data-group="group2" data-view="coronal" style="width: 100%; height: auto;">
                        <img src="data:image/png;base64,{sagittal_lesion_overlap_filtered_mosaic}" alt="Lesion Overlap Filtered Sagittal" data-group="group2" data-view="sagittal" style="width: 100%; height: auto;">
                    </div>
                    """)
        f.write(f"""
            </div>
            
            <div class="section">
                <h2>Model Parameters</h2>
                <table class="result-table">
                    <tr><th>Parameter</th><th>Value</th></tr>
            """)
        for param, value in svr_params.items():
            f.write(f"<tr><td>{param}</td><td>{value}</td></tr>\n")

        f.write(f"""
                </table>
                <div class="image-category">
                    <h3>Beta Map</h3>
                    <img src="data:image/png;base64,{axial_svr_beta_map_mosaic}" alt="Beta Map Axial" data-group="group3" data-view="axial" style="width: 100%; height: auto;" class="active">
                    <img src="data:image/png;base64,{coronal_svr_beta_map_mosaic}" alt="Beta Map Coronal" data-group="group3" data-view="coronal" style="width: 100%; height: auto;">
                    <img src="data:image/png;base64,{sagittal_svr_beta_map_mosaic}" alt="Beta Map Sagittal" data-group="group3" data-view="sagittal" style="width: 100%; height: auto;">
                </div>
                
                <h3>Permutation Tested<h3>
                <!-- Second Set of Radio Buttons (SVR Switcher) -->
                <div class="radio-buttons">
                    <label>
                        <input type="radio" name="svrSwitcher" value="axial" onclick="switchSvrView('axial');" checked>
                        <div class="radio-button-block">Axial</div>
                    </label>
                    <label>
                        <input type="radio" name="svrSwitcher" value="coronal" onclick="switchSvrView('coronal');">
                        <div class="radio-button-block">Coronal</div>
                    </label>
                    <label>
                        <input type="radio" name="svrSwitcher" value="sagittal" onclick="switchSvrView('sagittal');">
                        <div class="radio-button-block">Sagittal</div>
                    </label>
                </div>
            
                <!-- Third Set of Radio Buttons (Threshold Switcher) -->
                <div class="radio-buttons">
                    <label>
                        <input type="radio" name="thresholdSwitcher" value="unthresholded" onclick="toggleThresholdGroup('unthresholded')" checked>
                        <div class="radio-button-block">Unthresholded</div>
                    </label>
                    <label>
                        <input type="radio" name="thresholdSwitcher" value="p05" onclick="toggleThresholdGroup('p<0.05')">
                        <div class="radio-button-block">p<0.05</div>
                    </label>
                    <label>
                        <input type="radio" name="thresholdSwitcher" value="p01" onclick="toggleThresholdGroup('p<0.01')">
                        <div class="radio-button-block">p<0.01</div>
                    </label>
                    <label>
                        <input type="radio" name="thresholdSwitcher" value="p005" onclick="toggleThresholdGroup('p<0.005')">
                        <div class="radio-button-block">p<0.005</div>
                    </label>
                    <label>
                        <input type="radio" name="thresholdSwitcher" value="p001" onclick="toggleThresholdGroup('p<0.001')">
                        <div class="radio-button-block">p<0.001</div>
                    </label>
                </div>
            
                <!-- Z Map Group (4th div) -->
                <div class="image-category">
                    <h3>Z Map (Thresholded)</h3>
        
                    <!-- Unthresholded Images -->
                    <div class="threshold-group" data-threshold="unthresholded" style="display: block;">
                        <img src="data:image/png;base64,{axial_zmap_mosaic}" alt="Z Map Axial" data-group="group4" data-view="axial" style="width: 100%; height: auto;" class="active">
                        <img src="data:image/png;base64,{coronal_zmap_mosaic}" alt="Z Map Coronal" data-group="group4" data-view="coronal" style="width: 100%; height: auto;">
                        <img src="data:image/png;base64,{sagittal_zmap_mosaic}" alt="Z Map Sagittal" data-group="group4" data-view="sagittal" style="width: 100%; height: auto;">
                    </div>
        
                    <!-- Thresholded Images -->
                """)
        
        thresholds = [
            ('p05', 'p<0.05', 5),
            ('p01', 'p<0.01', 6),
            ('p005', 'p<0.005', 7),
            ('p001', 'p<0.001', 8)
        ]
        
        for threshold_suffix, threshold_label, group_num in thresholds:
            f.write(f'''
                    <div class="threshold-group" data-threshold="{threshold_label}" style="display: none;">
                        <img src="data:image/png;base64,{mosaics[f'axial_zmap_{threshold_suffix}']}" alt="Z Map {threshold_label} Axial" data-group="group{group_num}" data-view="axial" style="width: 100%; height: auto;" class="active">
                        <img src="data:image/png;base64,{mosaics[f'coronal_zmap_{threshold_suffix}']}" alt="Z Map {threshold_label} Coronal" data-group="group{group_num}" data-view="coronal" style="width: 100%; height: auto;">
                        <img src="data:image/png;base64,{mosaics[f'sagittal_zmap_{threshold_suffix}']}" alt="Z Map {threshold_label} Sagittal" data-group="group{group_num}" data-view="sagittal" style="width: 100%; height: auto;">
                    </div>
            ''')
            
        f.write(f"""
                </div>
            </div>
            """)

        if correlation_matrix_path.exists():
            correlation_matrix_mosaic = encode_image(correlation_matrix_path)
            f.write(f"""
            <div class="section">
                <h2>Covariate Correlations</h2>
                <p>Pearson correlation between the behavioural score (one-hot per class for
                    multiclass) and each covariate, computed before covariate regression.</p>
                <img src="data:image/png;base64,{correlation_matrix_mosaic}" alt="Behaviour/Covariate Correlation Matrix" style="max-width: 100%; height: auto;">
            </div>
            """)

        f.write(f"""
            <div class="section highlight">
                <h3>Results</h3>
                <ul>
                    <li><strong>Mean lesion volume (in voxels):</strong> {mean_lesion_volume:.2f}</li>
                    <li><strong>Number of lesion files:</strong> {num_lesions}</li>
                    <li><strong>Z-map value range:</strong> ({zmap_min:.2f}, {zmap_max:.2f})</li>
                </ul>
            </div>
            <div id="section">
                <h2>Interactive Viewer</h2>
                <div>
                    {html_view}
                </div>
            </div>

        """)


        f.write(f"""
                <div class="section">
                    <h3>Download</h3>
                    <p>Download the <a href='zmap.nii.gz'>Z-map NIfTI file</a> for further analysis.</p>
                </div>
            """)

        f.write("""</table></div>

            <div class="footer">
                <p>Generated by svmLSMpy (Support Vector Lesion-Symptom Mapping)</p>
            </div>

            <!-- Lightbox script -->
            <script>
                function openLightbox(event, element) {
                    event.preventDefault();  // Prevent the default anchor click action
                    var lightbox = document.createElement('div');
                    lightbox.classList.add('lightbox');
                    var img = document.createElement('img');
                    img.src = element.href;
                    lightbox.appendChild(img);
                    document.body.appendChild(lightbox);

                    // Add an event listener to close the lightbox when clicked
                    lightbox.addEventListener('click', function() {
                        lightbox.remove();
                    });
                }
            </script>
            </div>
        </body>
        </html>
        """)

    print(f"Report successfully saved to {output_file}.")











