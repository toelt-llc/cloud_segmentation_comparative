import requests
import os
from bs4 import BeautifulSoup
from config import *


def main():
    # Define the URL of the webpage
    url = "https://landsat.usgs.gov/landsat-8-cloud-cover-assessment-validation-data"

    # Define the list of biomes
    biomes = ['Barren', 'Forest', 'GrassCrops', 'Shrubland', 'SnowIce', 'Urban', 'Water', 'Wetlands']

    # Create the folders for each biome
    for biome in biomes:
        os.makedirs(f"{biome_raw_dir}/{biome}", exist_ok=True)

    # Send a GET request to the webpage
    response = requests.get(url)

    # Create a BeautifulSoup object to parse the HTML content
    soup = BeautifulSoup(response.text, "html.parser")

    # Find all the <a> tags that contain the download links
    links = soup.find_all("a")

    # Create a list to store the extracted links
    download_links = []

    # Extract the href attribute from each <a> tag
    for link in links:
        href = link.get("href")
        if href and href.endswith(".tar.gz"):  # Modify the condition based on your specific requirements
            download_links.append(href)

    # # Generate the .sh file with the wget and tar commands
    # with open("cloud_coverage_TOELT_SUPSI/src/download_landsat8_biome.sh", "w") as file:
    #     file.write("#!/bin/bash\n")
    #     file.write("# Auto-generated script to download and extract files\n")
    #     file.write("\n")
    #     # file.write("mkdir -p /path/to/extracted/files\n")  # Replace '/path/to/extracted/files' with the desired extraction folder
    #     file.write("\n")
    #     for link in download_links:
    #         file.write(f"wget {link}\n")
    #         filename = link.split("/")[-1]
    #         file.write(f"tar zxvf {filename} -C /home/floddo/cloud_coverage_TOELT_SUPSI/Data/L8_Biome\n")
    #         file.write(f"rm {filename}\n")  # Optional: Remove the downloaded .tar.gz file after extraction


    # Generate the .sh file with the wget and tar commands
    with open("cloud_coverage_TOELT_SUPSI/src/download_landsat8_biome.sh", "w") as file:
        file.write("#!/bin/bash\n")
        file.write("# Auto-generated script to download and extract files\n")
        file.write("\n")

        # Initialize a variable to keep track of the current biome index
        biome_index = 0

        # Add a loop to iterate over the download links
        for i, link in enumerate(download_links):
            # Check if the current link index is divisible by 12
            if i % 12 == 0 and i > 0:
                # If divisible by 12, update the biome index
                biome_index += 1

            # Get the current biome based on the biome index
            current_biome = biomes[biome_index % len(biomes)]

            file.write(f"wget {link}\n")
            filename = link.split("/")[-1]
            file.write(f"tar zxvf {filename} -C {biome_raw_dir}/{current_biome}\n")
            file.write(f"rm {filename}\n")  # Optional: Remove the downloaded .tar.gz file after extraction



if __name__ == "__main__":
    main()