import requests
import pandas as pd
import json
import os
import subprocess
import tqdm
import time

date_from = '2018'
date_to = '2019'

sentinel_product_meta_file = rf"output_folder/sentinel_download_python/sentinel_meta_{date_from}-{date_to}.csv"
if not os.path.isfile(sentinel_product_meta_file):
    response = requests.get(f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=contains(Name,'T33TYM') and contains(Name,'MSIL2') and ContentDate/Start gt {date_from}-01-01T00:00:00.000Z and ContentDate/Start lt {date_to}-01-01T00:00:00.000Z&$top=500")
    df = pd.DataFrame.from_dict(response.json()['value'])
    df.to_csv(sentinel_product_meta_file)

df = pd.read_csv(sentinel_product_meta_file)

df_tile = df[df['Name'].str.contains('_T33TYM_')]
df_msil1c = df_tile[df_tile['Name'].str.contains('_MSIL1C_')]
df_msil2a = df_tile[df_tile['Name'].str.contains('_MSIL2A_')]

df_diff = df_msil1c[~df_msil1c['ContentDate'].isin(df_msil2a['ContentDate'])]
df_filtered = pd.concat([df_diff, df_msil2a], ignore_index=True)
df_filtered['ProcessVersion'] = df_filtered['Name'].str.split('_').str[3].str[1:].astype('int64')
df_filtered_sorted = df_filtered.sort_values('ProcessVersion')

df_filtered = df_filtered_sorted.drop_duplicates('ContentDate', keep='last')
dfc = df_filtered.sort_values('ContentDate')

def get_token():
    print('Getting new token')
    curl_output = subprocess.check_output(r'./curl_get_token.bat')
    # if on linux
    # curl_output = subprocess.check_output([/bin/sh, os.path.abspath(r'curl_get_token.sh')])
    token_json = json.loads(curl_output)
    token = token_json['access_token']
    return token, time.time()


def get_filepath(i):
    # Replace with output folder
    export_folder = os.path.join(r'your_output_folder', date_from)
    if not os.path.exists(export_folder):
        os.makedirs(export_folder)
    filepath = os.path.join(export_folder, f"{df_to_download.iloc[i]['Name']}.zip")
    if os.path.exists(filepath):
        filesize = os.path.getsize(filepath)
        print(f"{filepath} already exists with size {filesize/1024} Kb")
        if filesize < df_to_download.iloc[i]['ContentLength']:
            print(f"File on disk is ({df_to_download.iloc[i]['Name']},"
                  f" {filesize/1024} kb) smaller than df ContentLength ("
                  f"{df_to_download.iloc[i]['ContentLength']/1024} kb), delete this manually!")
        return None
    return filepath


df_to_download = df_filtered.copy(deep=True)

for i in range(len(df_to_download.index)):
    filepath = get_filepath(i)
    if not filepath:
        continue

    token, token_time = get_token()
    url = (f"https://zipper.dataspace.copernicus.eu"
           f"/odata/v1/Products({df_to_download.iloc[i]['Id']})/$value")
    headers = {"Authorization": f"Bearer {token}"}

    print(f"[{i}/{len(df_to_download)}] Getting {df_to_download.iloc[i]['Name']}")
    session = requests.Session()
    session.headers.update(headers)

    tries = 5
    for n in range(tries):
        try:
            with session.get(url, headers=headers, stream=True) as response:
                # Sizes in bytes.
                total_size = int(response.headers.get("content-length", 0))

                if total_size < df_to_download.iloc[i]['ContentLength']:
                    print(f"total_size not matching previous result, skipping for now...")
                else:

                    block_size = 1024

                    with tqdm.tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
                        with open(filepath, "wb") as file:
                            for chunk in response.iter_content(chunk_size=1024 * 1024):
                                if chunk:
                                    progress_bar.update(len(chunk))
                                    file.write(chunk)
                            print(f"Download successful - {df_to_download.iloc[i]['Name']}")
                    if total_size != 0 and progress_bar.n != total_size:
                        raise RuntimeError("Could not download file")
                break
        except requests.exceptions.ChunkedEncodingError as e:
            print(e)
            print(f"Deleting broken file {filepath}")
            os.remove(filepath)
            print(f"Retry {n+1}, retrying...token_time:{time.time()-token_time} secs")
            time.sleep(10)

            if time.time()-token_time > 550:
                token, token_time = get_token()
            if n == tries - 1:
                raise e


