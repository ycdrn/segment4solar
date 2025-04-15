from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from werkzeug.utils import secure_filename
import requests
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import pickle
import matplotlib
matplotlib.use('Agg')
import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt
import io
import base64
import calendar
import matplotlib.font_manager as fm


app = Flask(__name__)

# Set the upload folder
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

# Allowed extensions for uploaded files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """Check if the file has one of the allowed extensions."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    error = None  # Initialize the error message
    if request.method == 'POST':
        # Check if the 'facadeImage' key is in the request.files dictionary
        if 'facadeImage' not in request.files:
            error = 'No file part in the request'
        else:
            file = request.files['facadeImage']
            # If the user does not select a file
            if file.filename == '':
                error = 'No file selected'
            if file and allowed_file(file.filename):
                # Secure the filename and save the file
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                # Redirect to the segmentation result page
                return redirect(url_for('show_segmentation', filename=filename))
            else:
                error = 'Invalid file type'
    # Render the template with any error messages
    return render_template('index.html', error=error)


# SEGMENT ANYTHING
# Details are here: https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints

# Path to the model checkpoint
MODEL_CHECKPOINT_PATH = os.path.join(app.root_path, 'models', 'sam_vit_b_01ec64.pth')

# Select the model type (e.g., "vit_h", "vit_l", "vit_b")
MODEL_TYPE = "vit_b"

print("Loading SAM model...")
# MPS doesn't work, that's why everything here is commented out
#device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
sam = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_CHECKPOINT_PATH) #.to(device)
mask_generator = SamAutomaticMaskGenerator(sam)
print("SAM model loaded successfully")


def segment_image(image_path):
    """
    Perform image segmentation using SAM.
    Returns masks and the image dimensions.
    """
    try:
        print(f"Loading image for segmentation: {image_path}")
        image = np.array(Image.open(image_path))
        print("Image loaded")

        # Initialize the mask generator
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=4,  # Reduce for faster processing
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=0,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=5  # Ignore very small regions
        )

        # Generate masks (list of masks)
        print("Starting mask generation...")
        masks_list = mask_generator.generate(image)
        print(f"Number of masks generated: {len(masks_list)}")

        # Save masks list for later use
        masks_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'masks.pkl')
        with open(masks_file_path, 'wb') as f:
            pickle.dump(masks_list, f)
        print(f"Masks list saved to {masks_file_path}")

        # Create an image with masks overlaid for display
        segmented_image_filename = 'segmented_image.png'  # Changed to PNG for consistency
        create_interactive_segmented_image(image, masks_list, segmented_image_filename)
        print(f"Segmented image created: {segmented_image_filename}")

        return segmented_image_filename, image.shape

    except Exception as e:
        print(f"An error occurred during segmentation: {e}")
        return None, None


def create_interactive_segmented_image(image, masks, output_filename):
    """
    Creates an SVG image with segmentation masks overlaid.
    Saves the SVG image for user interaction with hover effects.
    """
    print("Creating interactive segmented image (SVG) with hover effects...")
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage import measure
    import io
    import os

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image)

    # For each mask, create a Path and add it to the plot
    for idx, mask in enumerate(masks):
        if 'segmentation' not in mask:
            print(f"Mask at index {idx} does not contain 'segmentation' key. Skipping.")
            continue
        m = mask['segmentation']
        contours = measure.find_contours(m, 0.5)
        for contour in contours:
            ax.fill(
                contour[:, 1], contour[:, 0],
                facecolor=np.random.rand(3,),
                edgecolor='none',
                alpha=0.5,
                gid=f"mask_{idx}"
            )

    ax.axis('off')
    plt.tight_layout(pad=0)

    # Save the figure to an SVG buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='svg', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    svg_data = buf.getvalue().decode('utf-8')

    # Modify the SVG data to replace 'gid' with 'id' and add class 'mask-path'
    svg_data = svg_data.replace('gid="mask_', 'id="mask_')
    svg_data = svg_data.replace('style="', 'class="mask-path" style="')

    # Save the SVG data to a file
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    with open(output_path, 'w') as f:
        f.write(svg_data)

    print("Interactive segmented image (SVG) with hover effects created.")   


@app.route('/segmentation/<filename>')
def show_segmentation(filename):
    print(f"Starting segmentation for file: {filename}")
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    print(f"File path: {file_path}")

    # Perform segmentation
    segmented_image_filename, image_shape = segment_image(file_path)
    if segmented_image_filename is None:
        return "An error occurred during segmentation.", 500

    print("Segmentation completed")

    # Read the SVG content
    svg_path = os.path.join(app.config['UPLOAD_FOLDER'], segmented_image_filename)
    with open(svg_path, 'r') as svg_file:
        svg_content = svg_file.read()

    # Render the template to select the wall mask
    return render_template(
        'select_wall.html',
        original_image=filename,
        segmented_image=segmented_image_filename,
        svg_content=svg_content  # Pass the SVG content to the template
    )
 
 
@app.route('/process_selection/<filename>', methods=['POST'])
def process_selection(filename):
    # Retrieve form data at the beginning
    selected_mask_id = request.form.get('selected_mask_id')
    facade_area = request.form.get('facade_area')
    orientation = request.form.get('orientation')
    pv_technology = request.form.get('pv_technology')
    location = request.form.get('location')

    print(f"Received selected_mask_id: '{selected_mask_id}'")
    print(f"Received facade_area: '{facade_area}'")
    print(f"Received orientation: '{orientation}'")
    print(f"Received pv_technology: '{pv_technology}'")
    print(f"Received location: '{location}'")

    # Validate inputs
    if not selected_mask_id or not facade_area or not orientation or not pv_technology or not location:
        return "Please provide all required inputs.", 400

    try:
        selected_mask_id = int(selected_mask_id)
        facade_area = float(facade_area)
    except ValueError:
        return "Invalid input. Please enter valid numbers.", 400

    # Load masks
    masks_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'masks.pkl')
    if not os.path.exists(masks_file_path):
        return "Masks data not found.", 500

    with open(masks_file_path, 'rb') as f:
        masks = pickle.load(f)

    # Validate that masks is a list
    if not isinstance(masks, list):
        return "Internal server error: Invalid masks data format.", 500

    # Validate selected_mask_id exists in masks
    # Assuming mask IDs start from 0
    if selected_mask_id < 0 or selected_mask_id >= len(masks):
        return "Selected mask ID does not exist.", 400

    # Get the selected wall mask (0-based indexing)
    wall_mask = masks[selected_mask_id].get('segmentation')
    if wall_mask is None:
        return "Selected mask does not contain segmentation data.", 400

    # Calculate wall mask area
    wall_area = np.sum(wall_mask)
    print(f"Selected Mask ID: {selected_mask_id}")
    print(f"Wall Area (sum of wall_mask): {wall_area}")

    if wall_area == 0:
        return "Selected mask has zero area.", 400

    # Initialize obstructed area
    obstructed_area = np.zeros_like(wall_mask, dtype=bool)

    # Find smaller masks within the wall mask
    overlapping_masks = []
    for idx, mask in enumerate(masks):
        if idx == selected_mask_id:
            continue  # Skip the wall mask itself
        # Check if the mask overlaps with the wall mask
        mask_segmentation = mask.get('segmentation')
        if mask_segmentation is None:
            continue  # Skip masks without segmentation data

        overlap = np.logical_and(wall_mask, mask_segmentation)
        overlap_area = np.sum(overlap)
        mask_area = np.sum(mask_segmentation)
        overlap_ratio = overlap_area / mask_area if mask_area != 0 else 0
        print(f"Mask ID {idx}: Overlap Area = {overlap_area}, Mask Area = {mask_area}, Overlap Ratio = {overlap_ratio:.2f}")

        # Define a threshold for meaningful overlap, e.g., at least 10% overlap
        if overlap_ratio < 0.1:
            print(f"Mask ID {idx} has negligible overlap with the wall mask. Skipping.")
            continue  # Skip masks with negligible overlap

        # Mark overlapping areas as obstructed
        obstructed_area = np.logical_or(obstructed_area, overlap)
        overlapping_masks.append(idx)
        print(f"Mask ID {idx} overlaps with the wall mask.")

    # Compute the unobstructed wall mask
    unobstructed_wall_mask = np.logical_and(wall_mask, np.logical_not(obstructed_area))

    # Calculate unobstructed wall area
    unobstructed_area = np.sum(unobstructed_wall_mask)
    unobstructed_ratio = unobstructed_area / wall_area if wall_area != 0 else 0
    unobstructed_facade_area = unobstructed_ratio * facade_area

    print(f"Obstructed Area (sum of obstructed_area): {np.sum(obstructed_area)}")
    print(f"Unobstructed Area (sum of unobstructed_wall_mask): {unobstructed_area}")
    print(f"Unobstructed Ratio: {unobstructed_ratio}")
    print(f"Unobstructed Facade Area: {unobstructed_facade_area} square meters")

    # Create an image showing the unobstructed wall area
    try:
        unobstructed_image_filename = create_unobstructed_wall_image(filename, unobstructed_wall_mask)
        print(f"Unobstructed Image Filename: {unobstructed_image_filename}")
    except Exception as e:
        print(f"Error in creating unobstructed wall image: {e}")
        return f"Failed to create unobstructed wall image: {str(e)}", 500

    # Calculate energy yield and extract additional PVGIS data
    try:
        print(f"Calculating energy yield with:")
        print(f"Area: {unobstructed_facade_area}")
        print(f"Orientation: {orientation}")
        print(f"PV Technology: {pv_technology}")
        print(f"Location: {location}")

        energy_yield, pvgis_data, monthly_energy_plot = calculate_energy_yield(
            unobstructed_facade_area,
            orientation,
            pv_technology,
            location
        )
        print(f"Energy yield calculated: {energy_yield}")
    except Exception as e:
        print(f"An error occurred during energy yield calculation: {e}")
        energy_yield = None
        pvgis_data = {}
        monthly_energy_plot = ""

    # Render the results
    return render_template(
    'results.html',
    original_image=filename,
    selected_mask_id=selected_mask_id,  # Ensure this is passed
    orientation=orientation,            # Pass orientation
    pv_technology=pv_technology,        # Pass PV technology
    location=location,                  # Pass location
    unobstructed_ratio=unobstructed_ratio,
    unobstructed_facade_area=unobstructed_facade_area,
    unobstructed_image=unobstructed_image_filename,
    energy_yield=energy_yield,
    pvgis_data=pvgis_data,
    monthly_energy_plot=monthly_energy_plot
)

    

def create_unobstructed_wall_image(original_filename, unobstructed_wall_mask):
    """
    Creates an image highlighting the unobstructed wall area.
    Saves the image and returns the filename.
    
    Parameters:
        original_filename (str): Filename of the original image.
        unobstructed_wall_mask (numpy.ndarray): Boolean mask of unobstructed areas.
        
    Returns:
        str: Filename of the saved unobstructed wall image.
    """
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Original image not found: {image_path}")
    
    image = Image.open(image_path).convert("RGBA")
    image_width, image_height = image.size

    # Make sure mask shape matches image dimensions
    if unobstructed_wall_mask.shape != (image_height, image_width):
        unobstructed_wall_mask = unobstructed_wall_mask.transpose()

    # Create an overlay with semi-transparent green
    overlay_array = np.zeros((image_height, image_width, 4), dtype=np.uint8)
    overlay_color = (0, 255, 0, 100)  # Semi-transparent green (alpha=100)
    overlay_array[unobstructed_wall_mask] = overlay_color
    overlay = Image.fromarray(overlay_array, mode='RGBA')

    # Composite the overlay onto the original image
    segmented_image = Image.alpha_composite(image, overlay)

    # Save the image
    base_filename = os.path.splitext(original_filename)[0]
    segmented_filename = f'unobstructed_{base_filename}.png'
    segmented_image_path = os.path.join(app.config['UPLOAD_FOLDER'], segmented_filename)
    segmented_image.save(segmented_image_path, format='PNG')

    return segmented_filename


#def create_segmented_image(image_path, mask, original_filename):
    """
    Creates an image with the segmentation mask overlaid.
    Saves the image and returns the filename.
    """
    print("Creating segmented image...")
    image = Image.open(image_path).convert("RGBA")
    mask_array = mask.astype(np.uint8) * 255  # Convert boolean mask to uint8

    # Create an image from the mask
    mask_image = Image.fromarray(mask_array, mode='L')

    # Create a red overlay
    red_overlay = Image.new("RGBA", image.size, (255, 0, 0, 100))  # Semi-transparent red

    # Composite the overlay with the mask
    segmented_image = Image.composite(red_overlay, image, mask_image)

    segmented_filename = 'segmented_' + original_filename
    segmented_image_path = os.path.join(app.config['UPLOAD_FOLDER'], segmented_filename)
    segmented_image.save(segmented_image_path)
    print("Segmented image created and saved.")
    return segmented_filename



def calculate_energy_yield(area, orientation, pv_technology, location):
    """
    Calculate the potential annual energy yield based on the area, orientation, PV technology, and location.
    Utilizes PVGIS for accurate irradiance data and extracts additional simulation details.
    
    Parameters:
        area (float): Unobstructed facade area in square meters.
        orientation (str): Orientation of the wall (e.g., north, south).
        pv_technology (str): Type of PV technology (e.g., monocrystalline).
        location (str): City name.
        
    Returns:
        tuple: (energy_yield, pvgis_data, monthly_energy_plot)
            - energy_yield (float): Estimated annual energy yield in kilowatt-hours (kWh).
            - pvgis_data (dict): Additional PVGIS simulation data.
            - monthly_energy_plot (str): Base64-encoded PNG image of the monthly energy output graph.
    """
    # Step 1: Geocode the location to get latitude and longitude
    latitude, longitude = geocode_location(location)

    # Step 2: Fetch data from PVGIS
    pvgis_response = fetch_pvgis_data(latitude, longitude, orientation)

    # Step 3: Extract necessary data
    try:
        E_y = pvgis_response['outputs']['totals']['fixed']['E_y']  # Annual energy production per kWp
        monthly_data = pvgis_response['outputs']['monthly']['fixed']  # List of monthly data
    except KeyError as e:
        raise ValueError(f"Required PVGIS data missing: {e}")

    # Step 4: Retrieve the efficiency for the selected PV technology
    efficiency_data = {
        'monocrystalline': 0.20,
        'polycrystalline': 0.15,
        'thin_film': 0.10
    }
    
    efficiency = efficiency_data.get(pv_technology.lower())
    if not efficiency:
        raise ValueError(f"Unsupported PV technology: {pv_technology}")

    # Step 5: Calculate installed capacity
    installed_capacity = area * efficiency  # in kWp

    # Step 6: Calculate energy yield
    energy_yield = installed_capacity * E_y  # in kWh/year

    print(f"Energy Yield Calculation:")
    print(f"Area: {area} m²")
    print(f"Installed Capacity: {installed_capacity} kWp")
    print(f"Specific Energy Yield (E_y): {E_y} kWh/kWp/year")
    print(f"Energy Yield: {energy_yield} kWh/year")

    # Step 7: Process monthly energy data
    monthly_energy = {
        calendar.month_abbr[entry['month']]: entry['E_m'] * installed_capacity for entry in monthly_data
    }

    # Step 8: Generate Monthly Energy Output Graph
    monthly_energy_plot = generate_monthly_energy_plot(monthly_energy)

    # Step 9: Prepare additional PVGIS data
    pvgis_data = {
        'specific_energy_yield': E_y,
        'monthly_energy': monthly_energy,
        'simulation_inputs': pvgis_response['inputs']
    }

    return energy_yield, pvgis_data, monthly_energy_plot


def generate_monthly_energy_plot(monthly_energy):
    """
    Generates a bar chart of monthly energy outputs and returns it as a base64-encoded PNG image.
    
    Parameters:
        monthly_energy (dict): Dictionary with month abbreviations as keys and energy values as values.
        
    Returns:
        str: Base64-encoded PNG image of the plot.
    """
    # Set the desired font family
    desired_fonts = ['Helvetica Neue', 'Arial', 'sans-serif']
    
    # Check if 'Helvetica Neue' is available
    available_fonts = set([f.name for f in fm.fontManager.ttflist])
    selected_font = 'Arial'  # Default fallback
    
    for font in desired_fonts:
        if font in available_fonts:
            selected_font = font
            break
    
    plt.rcParams['font.family'] = selected_font
    f_size = 14
    
    # Extract month names and corresponding energy values
    month_names = list(monthly_energy.keys())
    energy_values = list(monthly_energy.values())
    
    plt.figure(figsize=(8, 5))  # Reduced figure size for better fit
    bars = plt.bar(month_names, energy_values, color='gray')
    plt.xlabel('Month', fontsize=f_size)
    plt.ylabel('Energy Output (kWh)', fontsize=f_size)
    plt.xticks(rotation=0, fontsize=f_size)
    plt.yticks(fontsize=f_size)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.1f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, -16),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=f_size)
    
    plt.tight_layout()
    
    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png', transparent=True)
    plt.close()
    img.seek(0)
    
    # Encode the image to base64
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    
    return plot_url


def geocode_location(location):
    
    """
    Geocode the location (city name) to obtain latitude and longitude using Nominatim API.
    
    Parameters:
        location (str): City name.
        
    Returns:
        tuple: (latitude, longitude)
    """
    geocode_url = "https://nominatim.openstreetmap.org/search"
    params = {
        'q': location,
        'format': 'json',
        'limit': 1
    }

    headers = {
        'User-Agent': 'Segment4Solar/1.0 (duran@arch.ethz.ch)' 
    }

    response = requests.get(geocode_url, params=params, headers=headers)

    if response.status_code != 200:
        raise ConnectionError(f"Nominatim API request failed with status code {response.status_code}")

    data = response.json()

    if not data:
        raise ValueError(f"Geocoding failed for location: {location}")

    latitude = float(data[0]['lat'])
    longitude = float(data[0]['lon'])

    return latitude, longitude



def fetch_pvgis_data(latitude, longitude, orientation):
    """
    Fetch irradiance data from PVGIS for the given location and orientation. Inputs can be found here:
    https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/getting-started-pvgis/api-non-interactive-service_en
    
    Parameters:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        orientation (str): Orientation of the wall (e.g., north, south).
        
    Returns:
        float: Annual total irradiance in kWh/m²/year.
        
    """
    
    # Map orientation to tilt and azimuth
    orientation_mapping = {
        'north': {'tilt': 90, 'azimuth': 180},
        'northeast': {'tilt': 90, 'azimuth': -135},
        'east': {'tilt': 90, 'azimuth': -90},
        'southeast': {'tilt': 90, 'azimuth': -45},
        'south': {'tilt': 90, 'azimuth': 0},
        'southwest': {'tilt': 90, 'azimuth': 45},
        'west': {'tilt': 90, 'azimuth': 90},
        'northwest': {'tilt': 90, 'azimuth': 135},
        }


    orientation_lower = orientation.lower()
    if orientation_lower not in orientation_mapping:
        raise ValueError(f"Invalid orientation: {orientation}")

    tilt = orientation_mapping[orientation_lower]['tilt']
    azimuth = orientation_mapping[orientation_lower]['azimuth']

    # PVGIS API endpoint
    pvgis_url = "https://re.jrc.ec.europa.eu/api/v5_2/PVcalc"

    # Parameters for the API
    params = {
        'lat': latitude,
        'lon': longitude,
        'outputformat': 'json',
        'userhorizon': '1',  # Default horizon
        'angle': tilt,       # Tilt angle
        'aspect': azimuth,   # Azimuth angle
        'loss': '14',        # System losses in %
        'peakpower': '1',    # Peak power of the PV system in kWp
        'tracking': '0',     # Tracking mode: 0 = no tracking
        'mountingplace': 'building', # Set mounting type to building integrated
    }

    try:
        response = requests.get(pvgis_url, params=params)
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"PVGIS API request failed: {e}")

    if response.status_code != 200:
        raise ConnectionError(f"PVGIS API request failed with status code {response.status_code}")

    try:
        data = response.json()
    except json.JSONDecodeError:
        raise ValueError("PVGIS API response is not valid JSON.")

    # **Debugging Step:** Print the entire response
    print("PVGIS API Response:")
    print(json.dumps(data, indent=4))

    # Check if there's an error in the response
    if 'error' in data:
        error_message = data['error'].get('message', 'Unknown error')
        raise ValueError(f"PVGIS API Error: {error_message}")

    return data


@app.route('/details')
def details():
    """
    Route to display the Calculation Details page.
    """
    return render_template('details.html')


print(app.url_map)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
