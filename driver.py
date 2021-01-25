from slicendice import *
from utils import *
from consts import *


# preprocessing method
def preprocess_advertiser_data(entity_df, view_set, entity_id_field, min_attribute_length=0):
    """
    :param entity_df: original, unprocessed dataframe with advertiser data (CSV string per cell "a,b,c")
    :param view_set: list of view identifiers
    :param entity_id_field: field identifier for the entity id
    :param min_attribute_length: a minimum length for all values, anything lower is automatically considered a
                                 stopword (int) (default=0)
    :return: dataframe with shape (n_entities, n_views + 1) with each cell a set of values
    """

    # build stopwords dict.
    stopword_dict = process_stopwords_dict_from_file(file='stopwords/stopwords.txt', all_views=view_set)

    # prune stopword attribute values and convert to sets
    for view in view_set:
        if view not in stopword_dict:
            entity_df[view] = entity_df[view].apply(
                lambda blob: set() if (pd.isnull(blob) or blob == '') else set([w for w in blob.split(',')]))
        else:
            entity_df[view] = entity_df[view].apply(lambda blob: set() if (pd.isnull(blob) or blob == '') else set(
                [w for w in blob.split(',') if not is_stopword(w, stopword_dict[view], min_attribute_length)]))

    logging.info('Removed stopword values and preprocessed successfully.')
    logging.info('New dataframe shape: {}'.format(entity_df.shape))
    return entity_df


if __name__ == '__main__':
    # set logging config
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    # create log directory
    log_dir_path = 'logs/'
    images_dir_path = 'images/'
    values_dir_path = 'values/'
    
    for path in [log_dir_path, images_dir_path, values_dir_path]:
        create_directory(dir_path=path)
        delete_dir_files(dir_path=path)
    
    all_views = ['country', 'IP']  # specify relevant set of views to consider
    entity_id = 'entity_id'  # entity identifier field name
    
    # initialize job parameters
    example_job_params = SliceNDiceJobParams(
        entity_id_field=entity_id,  # string denoting entity identifier field name
        all_views=all_views,  # list of all field names to use as views
        max_entities=10,  # cap discovered groups to this many entities
        view_limit=2,  # only look for groups across this many views
        local_entity_whitelist_enabled=True,  # decide to use a local whitelist to avoid rediscovering same groups
        log_dir_path=log_dir_path  # where to write logs for seed generation
    )
    
    # ingest data
    json_data_path = 'example-data.json'
    
    # periodically pull fresh data (applicable for changing data sources) and re-initialize block discovery
    logging.info('Starting new block discovery process.')
    entity_data_orig = pd.read_json(json_data_path, lines=True)
    
    # generate dataframe: (number of entities, number of features + 1 for entity identifier)
    entity_data_proc = entity_data_orig.copy(deep=True)  # so that we can keep entity_data_orig clean for future
    entity_data_proc = preprocess_advertiser_data(entity_df=entity_data_proc,
                                                  view_set=example_job_params.ALL_VIEWS,
                                                  entity_id_field=example_job_params.ENTITY_ID_FIELD)
    
    # run SliceNDice algorithm and output results along the way
    slicer = SliceNDice(entity_data=entity_data_proc, job_params=example_job_params)
    slicer.parallelize_block_discovery(n_blocks_per_thread=1, n_threads=1)
    