from radiant_mlhub import Dataset
from config import *


def main():
    ds = Dataset.fetch('ref_cloud_cover_detection_challenge_v1')
    # for c in ds.collections:
    #     print(c.id)
    # print(s2_path)
    # ds.download(output_dir=s2_path)
    print(f'Title: {ds.title}')
    print(f'DOI: {ds.doi}')
    print(f'Citation: {ds.citation}')
    print('\nCollection IDs and License:')
    for collection in ds.collections:
        print(f'    {collection.id} : {collection.license}')

    # Download only the test data
    ds.download(output_dir=s2_path, collection_filter=dict(ref_cloud_cover_detection_challenge_v1_test_source=[], ref_cloud_cover_detection_challenge_v1_test_labels=[]))
    # ds.download(output_dir=s2_path, collection_filter=dict(ref_cloud_cover_detection_challenge_v1_test_source=[]))



if __name__ == '__main__':
    main()