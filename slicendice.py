import numpy as np
import uuid
import datetime
import matplotlib
import matplotlib.patches as mpatches
import networkx as nx

from random import *
from multiprocessing import Process
from itertools import combinations
from copy import deepcopy

from consts import *
from utils import *

matplotlib.use('Agg')


class SliceNDiceJobParams:
    """
    Used to capture key settings for a SliceNDice job.
    """

    def __init__(self, entity_id_field,
                 all_views, max_entities, view_limit, local_entity_whitelist_enabled, log_dir_path):
        self.ENTITY_ID_FIELD = entity_id_field
        self.ALL_VIEWS = all_views
        self.MAX_ENTITIES = max_entities
        self.VIEW_LIMIT = view_limit
        self.LOCAL_ENTITY_WHITELIST_ENABLED = local_entity_whitelist_enabled
        self.LOG_DIR_PATH = log_dir_path


class SliceNDice:
    """
    This class encapsulates methods used to run the SliceNDice algorithm introduced in the research paper
    "SliceNDice: Mining Suspicious Multi-attribute Entity Groups with Multi-view Graphs"

    Internal Attributes:
        self.data_dict (Dictionary): the above, converted to a dictionary for optimization purposes
        self.job_params.ALL_VIEWS: list of all columns in self.entity_data which should be used to measure suspiciousness
        self.tensor_stats: a dictionary that maps keys (view ID) to tuples indicated mass,density,volume for a view
        self.tensor_stats_noidf: the above, without IDF scoring of values
        self.term_idf: dictionary keyed on view identifier, with value a dictionary that maps attribute values to IDFs
    """

    def log_block_review_metadata(self, chosen_entities, chosen_views, block_metadata):
        """
        Log relevant assets.
        :param chosen_entities: set of chosen entities in the block to log
        :param chosen_views: list of chosen views found in the block
        :param block_metadata: dictionary describing block metadata, key is view identifier, and value is tuple of
            (view mass, view volume, view density, view value counts)
        :return:
        """

        image_directory = 'images/'
        value_directory = 'values/'

        # set block identifier
        block_uuid = str(uuid.uuid4())

        # get relevant block rows
        chosen_entities = chosen_entities
        chosen_views = chosen_views
        chosen_entity_rows = [row for row in self.data_dict.values() if
                              row[self.job_params.ENTITY_ID_FIELD] in chosen_entities]

        # rank views by most to least promising
        ranked_views = []
        for view in self.job_params.ALL_VIEWS:
            if block_metadata[view][2] > self.tensor_stats[view][2]:
                view_susp = self.calculate_susp_for_single_view(len(chosen_entities), block_metadata[view][0],
                                                                self.tensor_stats[view][2])
                ranked_views.append((view, view_susp))

        ranked_views.sort(key=lambda change: change[1], reverse=True)
        view_to_score_map = dict(ranked_views)
        ranked_view_colors = plt.get_cmap('gist_rainbow')(np.linspace(0, 1, len(self.job_params.ALL_VIEWS)))
        ranked_view_weights = np.linspace(5, 3, len(chosen_views)).tolist() + [0.1] * (
                len(self.job_params.ALL_VIEWS) - len(chosen_views))

        # create graph
        G = nx.MultiGraph()
        G.add_nodes_from(chosen_entities)
        for rank_id, (view, view_susp) in enumerate(ranked_views):
            edges = {}
            color = ranked_view_colors[rank_id]
            weight = ranked_view_weights[rank_id]

            for row in chosen_entity_rows:
                attribute_set = row[view]
                entity_id = row[self.job_params.ENTITY_ID_FIELD]
                for val in attribute_set:
                    if val not in edges:
                        edges[val] = set()
                    edges[val].add(entity_id)

            for val, entity_set in edges.items():
                if len(entity_set) < 2:
                    continue
                for pair in combinations(entity_set, 2):
                    G.add_edge(pair[0], pair[1], weight=weight, color=color, shared=(view, val))

        # visualize graph and save to file
        fig = plt.figure(figsize=(15, 15))
        edges = list(G.edges(data=True))
        edges.sort(key=lambda edge: edge[2]['weight'])
        edge_colors = []
        edge_weight = []
        for (u, v, attr_dict) in edges:
            edge_colors.append(attr_dict['color'])
            edge_weight.append(attr_dict['weight'])

        pos = nx.circular_layout(G)
        nx.draw(G, pos, edgelist=edges, edge_color=edge_colors, width=edge_weight, with_labels=True, font_size=9,
                node_size=100)
        edge_label_dict = build_edge_label_dict(nx.get_edge_attributes(G, 'shared'))
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_label_dict, font_size=6, bbox=dict(alpha=0))
        x_values, y_values = zip(*pos.values())
        x_min, x_max = min(x_values), max(x_values)
        x_margin = (x_max - x_min) * 0.25
        plt.xlim(x_min - x_margin, x_max + x_margin)
        y_min, y_max = min(y_values), max(y_values)
        y_margin = (y_max - y_min) * 0.25
        plt.xlim(x_min - x_margin, x_max + x_margin)
        plt.ylim(y_min - y_margin, y_max + y_margin)
        patches = []
        for rank_id, (view, view_susp) in enumerate(ranked_views):
            color = ranked_view_colors[rank_id]
            patches.append(mpatches.Patch(color=color, label='{} ({})'.format(view, view_to_score_map[view])))
        fig.legend(handles=patches, ncol=2)
        fig.show()

        graph_out_path = image_directory + block_uuid + '.pdf'
        fig.savefig(graph_out_path, format='pdf')
        plt.close(fig)
        logging.debug('Wrote graph for block {}'.format(block_uuid))

        # generate values asset file
        value_out_path = value_directory + block_uuid + '.txt'
        value_buf = open(value_out_path, 'w')
        value_buf.write('Number of entities: {}'.format(len(chosen_entity_rows)))

        for (view, _) in ranked_views:
            value_buf.write('\n\n\n----------------------------\n')
            value_buf.write(view + '\n')
            value_buf.write('----------------------------\n\n\n')
            counts = {}
            for row in chosen_entity_rows:
                for val in row[view]:
                    if val not in counts:
                        counts[val] = set()
                    counts[val].add(row[self.job_params.ENTITY_ID_FIELD])
            counts = list((val, len(entities), entities) for val, entities in counts.items() if len(entities) > 1)
            counts.sort(key=lambda val_tup: val_tup[1], reverse=True)
            for tup in counts:
                value_buf.write("\t{}, {}\n".format(tup[0], tup[1]))
                for entity in tup[2]:
                    value_buf.write("\t\t{}\n".format(entity))

        value_buf.close()
        logging.debug('Wrote values for block {}'.format(block_uuid))

        # log results to BQ
        block_susp = self.get_interpretable_susp_score(self.calculate_susp_of_block(chosen_views, block_metadata))
        block_size = len(chosen_entities)
        found_at = str(datetime.datetime.now())

        result_row = (block_uuid, block_size, block_susp, chosen_entities, chosen_views,
                      graph_out_path, value_out_path, found_at)
        logging.info('Row for block {}: {}'.format(block_uuid, result_row))

    def seed_views(self, percentile=99.5):
        """
        Sample views inversely proportional to a percentile of their empirical value frequency distributions.
        De-prioritizes "low-signal" views in which sharing frequency is very high, in favor of "high-signal" views.

        :param percentile: the percentile to use to evaluate the value frequency distribution (99.5 by default)
        :return: list of chosen views
        """

        views_to_sort = []
        for view in self.job_params.ALL_VIEWS:
            scores = []
            if self.remaining_tensor_stats[view][0] == 0:
                logging.debug('View {} fully exhausted (0 mass left).  Removing from seeding.'.format(view))
                continue  # view is fully sparse... avoid using it for seeding (it can never be satisfied)
            for key in self.tensor_stats_noidf[view][3]:
                scores.append(self.tensor_stats_noidf[view][3][key])
            views_to_sort.append({"view": view, "weight": 1.0 / np.percentile(scores, percentile)})

        if len(views_to_sort) < self.job_params.VIEW_LIMIT:
            return UNABLE_TO_FIND_BLOCK  # not enough dense views to find something according to specification

        views_to_sort.sort(key=lambda tup: tup["weight"], reverse=True)

        selection = pd.DataFrame(views_to_sort)
        selected = selection.sample(n=self.job_params.VIEW_LIMIT, weights=selection["weight"])

        return_list = list(selected["view"])
        return return_list

    def seed_entities(self, chosen_views, local_entity_whitelist=set()):
        """
        Select entities in seed using a stochastic algorithm, given a list of views.

        :param local_entity_whitelist: set of entities to avoid
        :param chosen_views: list of chosen views over which density constraints should be satisfied.
        :return: set of chosen entities which satisfy density constraints over the list of views provided.
        """

        num_iters = 0
        seed_found = False
        while not seed_found:
            if num_iters > 1000:
                logging.debug('Unable to grow a suitable seed on these views.')
                return UNABLE_TO_FIND_BLOCK
            num_iters += 1

            chosen_entities = set()

            # get the initial two entities that match on some view
            num_attempts_to_initialize_seed = 0
            while True:
                if num_attempts_to_initialize_seed > 1000:
                    logging.debug('Unable to initialize a seed on these views.')
                    return UNABLE_TO_FIND_BLOCK  # cannot initialize the seed
                initial_view = choice(chosen_views)
                logging.debug('Chose initial view {}'.format(initial_view))
                all_lists_of_connected_entities = list(self.tensor_stats_noidf[initial_view][5].values())
                chosen_list_of_connected_entities = set(choice(all_lists_of_connected_entities))
                connected_entity_candidates = chosen_list_of_connected_entities - local_entity_whitelist
                if len(connected_entity_candidates) < 2:
                    logging.debug('Chosen value list not long enough, trying again...'.format(initial_view))
                    num_attempts_to_initialize_seed += 1
                    continue
                chosen_entities = set(sample(connected_entity_candidates, 2))
                logging.debug('Chose entities: {}'.format(chosen_entities))
                break

            # get block stats
            block_stats = self.compute_block_metadata(chosen_entities)

            # check constraints
            unsatisfied_views = []
            for view in chosen_views:
                if block_stats[view][2] < self.tensor_stats[view][2]:
                    unsatisfied_views.append(view)

            logging.debug('Unsatisfied views: {}'.format(unsatisfied_views))

            if not len(unsatisfied_views):
                logging.debug('No unsatisfied views! Breaking.'.format(unsatisfied_views))
                seed_found = True
                break

            # grow and try to satisfy constraints
            shuffle(unsatisfied_views)
            seed_satisfiable = True

            for view_to_satisfy in unsatisfied_views:
                logging.debug('Trying to satisfy constraint for view {}'.format(view_to_satisfy))
                view_constraint_satisfied = False
                all_lists_of_connected_entities = list(self.tensor_stats_noidf[view_to_satisfy][5].values())
                shuffle(all_lists_of_connected_entities)
                shuffled_chosen_entities = list(chosen_entities)
                shuffle(shuffled_chosen_entities)
                for entity in shuffled_chosen_entities:
                    for list_of_connected_entities in all_lists_of_connected_entities:
                        if entity in list_of_connected_entities:
                            candidate_additions = [cand_entity for cand_entity in list_of_connected_entities if
                                                   cand_entity != entity and cand_entity not in local_entity_whitelist]
                            if len(candidate_additions):
                                entity_to_add = choice(candidate_additions)
                                chosen_entities.add(entity_to_add)
                                block_stats = self.compute_block_metadata(chosen_entities)
                                if block_stats[view_to_satisfy][2] > self.tensor_stats[view_to_satisfy][2]:
                                    logging.debug('View constraint satisfied by adding {}'.format(entity_to_add))
                                    view_constraint_satisfied = True
                                    break
                                else:
                                    logging.debug(
                                        'View constraint still not satisfied by adding {}'.format(entity_to_add))

                    if view_constraint_satisfied:
                        logging.debug('View constraint satisfied so moving on to next view')
                        break

                if not view_constraint_satisfied:
                    logging.debug('Tried everything but cant satisfy this view!  Seed is dead')
                    seed_satisfiable = False
                    break

            if not seed_satisfiable:
                logging.debug('Trying next seed iter: {}'.format(num_iters))
                continue
            else:
                seed_found = True
                logging.debug('Found a seed.  Validating...')
                for view in chosen_views:
                    if block_stats[view][2] < self.tensor_stats[view][2]:
                        seed_found = False
                        logging.debug('Seed did not meet density constraints')
                if len(chosen_entities) > self.job_params.MAX_ENTITIES:
                    seed_found = False
                    logging.debug('Seed size {} too large given max group capacity of {}'.format(len(chosen_entities),
                                                                                                 self.job_params.MAX_ENTITIES))
                if seed_found:
                    logging.debug('Seed validated!  On iter: {}'.format(num_iters))

        logging.debug('Final chosen entities: {}'.format(chosen_entities))
        return set(chosen_entities)

    def calculate_susp_for_single_view(self, num_nodes_in_block, mass_in_block, background_view_density):
        """
        Calculates suspiciousness score given block size, mass and background view density.

        :param num_nodes_in_block: n from the paper; number of nodes in the block being considered
        :param mass_in_block: m from the paper; total mass (sum of edge values) in the block being considered
        :param background_view_density: P from the paper; background density of the view being considered
        :return: suspiciousness score of a single view, defined as per Defn. 2 (MVSG scoring metric) from the paper;
        negative log-likelihood under MVERE model
        """
        volume = (num_nodes_in_block * (num_nodes_in_block - 1)) / 2.0
        susp = (volume * np.log(background_view_density)) + (volume * np.log(volume)) - volume - np.log(volume) - (
                volume * np.log(mass_in_block)) + np.log(mass_in_block) + (
                       mass_in_block * (1.0 / background_view_density))

        #return np.log2(susp)
        return susp

    def compute_block_metadata(self, chosen_entities, disable_idf=False, return_attr_to_entity_map=False):
        """
        Compute block metadata over chosen entities and all views.

        :param chosen_entities: set of chosen entities
        :param disable_idf: whether to use IDF for scoring mass.  if True, only use counts; if False, use IDF
        :param return_attr_to_entity_map: whether to return a mapping of attribute values to associated entities; if
        True, it's returned, and if False, it is not
        :return: dictionary describing block metadata; key is view identifier, and value is tuple of
        (view mass, view volume, view density, view value counts) + (optional attribute to entity map)
        """

        chosen_entity_rows = [entity_row for entity_row in self.data_dict.values() if
                              entity_row[self.job_params.ENTITY_ID_FIELD] in chosen_entities]
        block_metadata = {}

        # for each view
        for view in self.job_params.ALL_VIEWS:
            counts = {}  # count number of recurring values across every single org in entity_data
            attr_to_entity_map = {}  # mapping from attribute values to all entities which have that value

            for row in chosen_entity_rows:
                attribute_set = row[view]
                entity_id = row[self.job_params.ENTITY_ID_FIELD]

                for element in attribute_set:
                    if is_valid_attr_value(element):
                        if element not in counts:
                            counts[element] = 0
                            if return_attr_to_entity_map:
                                attr_to_entity_map[element] = set()
                        counts[element] += 1

                        if return_attr_to_entity_map:
                            attr_to_entity_map[element].add(entity_id)

            # compute mass, size, density, etc.
            if disable_idf:
                mass = sum(int(value_count) ** 2 - int(value_count) for value_count in counts.values())
            else:
                mass = sum(
                    self.term_idf[view][term] * ((value_count ** 2) - value_count) for term, value_count in
                    counts.items())
            size = float(len(chosen_entities))

            mass /= 2.0  # divide by half because we are concerned with edge sums, not the adjacency matrix sum
            volume = ((size * (size - 1)) / 2.0)
            density = mass / volume

            # store these inside of a dict, indexed by view id
            block_metadata[view] = (mass, size, density, counts)

            if return_attr_to_entity_map:
                block_metadata[view] = (mass, size, density, counts, attr_to_entity_map)

        return block_metadata

    def calculate_susp_of_block(self, chosen_views, block_metadata):
        """
        Calculates the suspiciousness of a block across chosen views.

        :param chosen_views: list of currently chosen views.
        :param block_metadata: dictionary describing block metadata, key is view identifier, and value is tuple of
        (view mass, view volume, view density, view value counts)
        :return: suspiciousness score of the block
        """
        susp = 0
        for view in chosen_views:
            susp += self.calculate_susp_for_single_view(block_metadata[view][1], block_metadata[view][0],
                                                        self.tensor_stats[view][2])

        susp /= float(len(chosen_views))
        return susp

    def compute_mass_delta(self, entity, chosen_views, chosen_entities, block_metadata):
        """
        Calculates the hypothetical change in mass by removing or adding an entity, across chosen views.

        :param entity: Pandas Series (or dictionary) for the entity in question.
        :param chosen_views: list of currently chosen views.
        :param chosen_entities: set of currently chosen entities.
        :param block_metadata: dictionary describing block metadata, key is view identifier, and value is tuple of
        (view mass, view volume, view density, view value counts)
        :return: dictionary of hypothetical change in mass, keyed by view
        """

        deltas = {}
        entity_id = entity[self.job_params.ENTITY_ID_FIELD]

        for view in chosen_views:
            counts, mass = block_metadata[view][3], block_metadata[view][0]
            delta_mass = 0
            # delta from removing an entity
            if entity_id in chosen_entities:
                for value in entity[view]:
                    if is_valid_attr_value(value):
                        current_contribution = (counts[value] ** 2) - counts[value]
                        new_contribution = (counts[value] - 1) ** 2 - (counts[value] - 1)
                        delta_mass += self.term_idf[view][value] * (new_contribution - current_contribution)
            # delta from adding an entity
            else:
                for value in entity[view]:
                    if value in counts:
                        current_contribution = counts[value] ** 2 - counts[value]
                        new_contribution = (counts[value] + 1) ** 2 - (counts[value] + 1)
                        delta_mass += self.term_idf[view][value] * (new_contribution - current_contribution)

            # Divide by half because we are concerned with edge sums, not the adjacency matrix sum
            deltas[view] = delta_mass / 2.0
        return deltas

    def compute_susp_delta(self, entity, chosen_views, chosen_entities, block_metadata):
        """
        Calculates the hypothetical change in suspiciousness by removing or adding an entity, across chosen views.

        :param entity: Pandas Series (or dictionary) for the entity in question.
        :param chosen_views: list of currently chosen views.
        :param chosen_entities: set of currently chosen entities.
        :param block_metadata: dictionary describing block metadata, key is view identifier, and value is tuple of
        (view mass, view volume, view density, view value counts)
        :return: hypothetical change in suspiciousness
        """

        delta_susp = 0
        entity_id = entity[self.job_params.ENTITY_ID_FIELD]
        mass_deltas = self.compute_mass_delta(entity, chosen_views, chosen_entities, block_metadata)

        num_chosen_entities = len(chosen_entities)

        for view in chosen_views:
            delta_mass = mass_deltas[view]
            delta_size = 1 if entity_id not in chosen_entities else -1

            projected_size = num_chosen_entities + delta_size
            projected_mass = block_metadata[view][0] + delta_mass
            projected_vol = float((projected_size * (projected_size - 1)) / 2.0)
            projected_density = projected_mass / projected_vol

            # if this change would violate a constraint, it cannot be beneficial
            if projected_density < self.tensor_stats[view][2]:
                return -np.inf

            new_susp = self.calculate_susp_for_single_view(num_chosen_entities + delta_size,
                                                           block_metadata[view][0] + delta_mass,
                                                           self.tensor_stats[view][2])
            old_susp = self.calculate_susp_for_single_view(num_chosen_entities, block_metadata[view][0],
                                                           self.tensor_stats[view][2])
            delta_susp += new_susp - old_susp

        # take the mean susp. over the views
        delta_susp /= float(len(chosen_views))
        return delta_susp

    def add_or_del_entity(self, entity, chosen_entities, block_metadata):
        """
        Add or delete an entity to/from the current chosen_entities (inplace) and update block metadata (inplace).
        Note that this *modifies the original* chosen_entities and block_metadata*!

        :param entity: Pandas Series (or dictionary) for the entity in question.
        :param chosen_entities: set of currently chosen entities (will be updated inplace).
        :param block_metadata: dictionary describing block metadata, key is view identifier, and value is tuple of
        (view mass, view volume, view density, view value counts)
        :return: metadata of the newly updated block
        """

        mass_deltas = self.compute_mass_delta(entity, self.job_params.ALL_VIEWS, chosen_entities, block_metadata)
        entity_id = entity[self.job_params.ENTITY_ID_FIELD]

        new_block_metadata = {}

        # update the entities list
        should_add_org = not (entity_id in chosen_entities)
        chosen_entities.add(entity_id) if should_add_org else chosen_entities.remove(entity_id)

        # update the value counts in all the views
        for view in self.job_params.ALL_VIEWS:
            entity_attr_values_for_view = entity[view]
            counts, mass, size = block_metadata[view][3], block_metadata[view][0], block_metadata[view][1]  # references
            if should_add_org:
                for attr_value in entity_attr_values_for_view:
                    if is_valid_attr_value(attr_value):
                        if attr_value not in counts:
                            counts[attr_value] = 0
                        counts[attr_value] += 1
            else:
                for attr_value in entity_attr_values_for_view:
                    if is_valid_attr_value(attr_value):
                        counts[attr_value] -= 1

            # update the properties
            mass += mass_deltas[view]
            size = (size + 1) if should_add_org else (size - 1)
            volume = ((size * (size - 1)) / 2.0)
            density = mass / volume
            new_block_metadata[view] = (mass, size, density, counts)
        return new_block_metadata

    def get_interpretable_susp_score(self, susp):
        """
        This is used to transform original suspiciousness score to something more interpretable.

        :param susp: original (untransformed) suspiciousness score
        :return: more interpretable (transformed) suspiciousness score
        """
        return susp

    def optimize_block_susp(self, chosen_entities, chosen_views, random_seed, local_entity_whitelist=set()):
        """
        Given a seed, run alternating maximization on entities and views to produce a block which is locally optimal in
        terms of suspiciousness score.

        :param chosen_entities: set of chosen entities in the seed
        :param chosen_views: list of chosen views in the seed
        :param random_seed: random seed identifier used to designate the block (and enable re-discovery for testing).
        :param local_entity_whitelist: set of entities to exclude from consideration for this block discovery process.
        :return: tuple with (set of chosen entities, set of chosen views, history of entity changes, and final
        suspiciousness score)
        """

        iteration = ENTITY_ITERATION
        block_metadata = self.compute_block_metadata(chosen_entities)
        history = [list(chosen_entities)]
        current_susp = self.calculate_susp_of_block(chosen_views, block_metadata)
        print_line_to_log_file('Starting with seed susp. {}'.format(self.get_interpretable_susp_score(current_susp)),
                               random_seed, self.job_params.LOG_DIR_PATH)

        while True:
            print_line_to_log_file("Starting {} iteration with susp. {}".format(iteration,
                                                                                self.get_interpretable_susp_score(
                                                                                    current_susp)),
                                   random_seed, self.job_params.LOG_DIR_PATH)

            # Ranking step (for both entity and view iteration)
            ranked_changes = []
            if iteration == ENTITY_ITERATION:
                print_line_to_log_file('Starting with {} chosen entities'.format(len(chosen_entities)),
                                       random_seed, self.job_params.LOG_DIR_PATH)

                # Stop iterating if we've reached max capacity
                if len(chosen_entities) >= self.job_params.MAX_ENTITIES:
                    print_line_to_log_file('Maximum group capacity of {} entities reached'.format(self.job_params.MAX_ENTITIES),
                                           random_seed, self.job_params.LOG_DIR_PATH)
                    break

                # For entities already selected compute benefit of de-selecting them
                # For entities not selected, compute benefit of selecting them
                for row in self.data_dict.values():
                    entity_id = row[self.job_params.ENTITY_ID_FIELD]

                    if entity_id in local_entity_whitelist:
                        continue
                    ranked_changes.append(
                        (row, self.compute_susp_delta(row, chosen_views, chosen_entities, block_metadata)))

            else:
                print_line_to_log_file("Starting with chosen views: {}".format(chosen_views),
                                       random_seed, self.job_params.LOG_DIR_PATH)
                for view in self.job_params.ALL_VIEWS:
                    if block_metadata[view][2] > self.tensor_stats[view][2]:
                        ranked_changes.append(
                            (view, self.calculate_susp_for_single_view(len(chosen_entities), block_metadata[view][0],
                                                                       self.tensor_stats[view][2])))

            # rank changes by most to least promising
            ranked_changes.sort(key=lambda change: change[1], reverse=True)

            # Filtering step (only for entity iteration)
            if iteration == ENTITY_ITERATION:
                print_line_to_log_file("Filtering bad entity candidates.".format(iteration), random_seed,
                                       self.job_params.LOG_DIR_PATH)
                while len(ranked_changes) > 0:
                    next_change = ranked_changes[0]
                    benefit = next_change[1]

                    # remove no-benefit changes from list
                    if benefit <= 0:
                        ranked_changes.pop(0)
                        continue

                    # if we got here, we found a suitable candidate
                    break

                # if there is nothing to change about the entities, terminate the optimization
                if len(ranked_changes) == 0:
                    print_line_to_log_file('No entities left after filtering.', random_seed,
                                           self.job_params.LOG_DIR_PATH)
                    break

            #  Revision step (for both entity and view iteration)
            if iteration == ENTITY_ITERATION:
                print_line_to_log_file('Starting entity revision step.', random_seed, self.job_params.LOG_DIR_PATH)
                next_change = ranked_changes.pop(0)
                benefit = self.compute_susp_delta(next_change[0], chosen_views, chosen_entities, block_metadata)
                history.append(next_change[0][self.job_params.ENTITY_ID_FIELD])
                if benefit > 0:
                    block_metadata = self.add_or_del_entity(next_change[0], chosen_entities, block_metadata)
                    print_line_to_log_file('Top entity: {}'.format(next_change[0][self.job_params.ENTITY_ID_FIELD]),
                                           random_seed, self.job_params.LOG_DIR_PATH)
                    new_susp = self.calculate_susp_of_block(chosen_views, block_metadata)
                else:
                    break  # the revision would be unfruitful
            else:
                print_line_to_log_file("Starting view revision step.", random_seed, self.job_params.LOG_DIR_PATH)
                chosen_views = [x[0] for x in ranked_changes][:self.job_params.VIEW_LIMIT]
                print_line_to_log_file('Top views: {}'.format(chosen_views), random_seed, self.job_params.LOG_DIR_PATH)
                new_susp = np.mean([x[1] for x in ranked_changes][:self.job_params.VIEW_LIMIT])

            print_line_to_log_file(
                '{} chosen views and {} chosen entities.'.format(len(chosen_views), len(chosen_entities)),
                random_seed, self.job_params.LOG_DIR_PATH)

            # throw exception if susp (meaningfully, barring precision issues) decreases -- this should never happen.
            if new_susp <= current_susp and (not np.isclose(new_susp, current_susp)):
                print_line_to_log_file('{}: {} iteration decreased susp to {}'.format(random_seed, iteration, new_susp),
                                       random_seed, self.job_params.LOG_DIR_PATH)
                raise AssertionError('{}: {} iteration decreased susp'.format(random_seed, iteration))

            # update suspiciousness score
            current_susp = new_susp
            print_line_to_log_file("Completing {} iteration with susp. {}".format(iteration,
                                                                                  self.get_interpretable_susp_score(
                                                                                      current_susp)),
                                   random_seed, self.job_params.LOG_DIR_PATH)

            if iteration == ENTITY_ITERATION:
                iteration = VIEW_ITERATION
            else:
                iteration = ENTITY_ITERATION

            print_line_to_log_file('\n', random_seed, self.job_params.LOG_DIR_PATH)

        return chosen_entities, chosen_views, history, current_susp

    def find_one_block(self, random_seed, local_entity_whitelist=set()):
        """
        Randomly select seed entities, views, and run local optimization algorithm.

        :param random_seed: random seed identifier used to designate the block (and enable re-discovery for testing).
        :param local_entity_whitelist: set of entities to exclude from consideration for this block discovery process.
        :return: tuple with (set of chosen entities, set of chosen views, history of entity changes, and final
        suspiciousness score)
        """

        seed(random_seed)
        np.random.seed(random_seed)

        logging.debug(
            'Logging block generation steps to {}'.format(self.job_params.LOG_DIR_PATH + str(int(random_seed))))

        # choose views
        chosen_views = self.seed_views()
        print_line_to_log_file('Seed views:', random_seed, self.job_params.LOG_DIR_PATH)
        print_line_to_log_file(chosen_views, random_seed, self.job_params.LOG_DIR_PATH)
        if chosen_views == UNABLE_TO_FIND_BLOCK:
            return UNABLE_TO_FIND_BLOCK

        # choose entities
        chosen_entities = self.seed_entities(chosen_views, local_entity_whitelist)
        print_line_to_log_file('Seed entities:', random_seed, self.job_params.LOG_DIR_PATH)
        print_line_to_log_file(chosen_entities, random_seed, self.job_params.LOG_DIR_PATH)
        if chosen_entities == UNABLE_TO_FIND_BLOCK:
            return UNABLE_TO_FIND_BLOCK

        # grow seed
        result = self.optimize_block_susp(chosen_entities, chosen_views, random_seed, local_entity_whitelist)

        # write resulting block metadata to logfile
        print_object_to_log_file(result, random_seed, self.job_params.LOG_DIR_PATH)
        self.log_block_review_metadata(result[0], result[1], self.compute_block_metadata(result[0]))
        return result

    def find_multiple_blocks(self, random_seed, n_blocks):
        """
        This is run on each thread, and aims to find multiple blocks, given constraints.  This is achieved by calling
        "find_one_block" multiple times, sequentially.

        :param random_seed: Random seed identifier used to designate the block (and enable re-discovery for testing).
        :param n_blocks: how many blocks this thread should try to produce.
        this will prevent future blocks on this thread from overlapping with previous ones.
        :return:
        """

        found_blocks = []
        local_entity_whitelist = set()
        remaining_entities = set(self.all_entities)

        for block_id in range(n_blocks):
            logging.debug('Finding block_id {}, using localized whitelist: {}'.format(block_id,
                                                                                      list(local_entity_whitelist)))
            result = self.find_one_block(random_seed + block_id, local_entity_whitelist)
            if result != UNABLE_TO_FIND_BLOCK:
                found_blocks.append(result)
                if self.job_params.LOCAL_ENTITY_WHITELIST_ENABLED:
                    entities_in_block = result[0]
                    logging.debug('Found block with entities {}'.format(entities_in_block))
                    local_entity_whitelist.update(entities_in_block)  # add found entities to whitelist for next block
                    entities_in_block_rows = [row for row in self.data_dict.values() if
                                              row[self.job_params.ENTITY_ID_FIELD] in entities_in_block]
                    for entity_row in entities_in_block_rows:
                        #  delete entity from remaining entities and update remaining_tensor_stats
                        self.remaining_tensor_stats = self.add_or_del_entity(entity_row,
                                                                             remaining_entities,
                                                                             self.remaining_tensor_stats)
                        logging.debug('Deleted entity {}, with {} remaining entities.'.format(
                            entity_row[self.job_params.ENTITY_ID_FIELD],
                            len(remaining_entities)))

        return found_blocks

    def parallelize_block_discovery(self, n_blocks_per_thread=10, n_threads=1):
        """
        Starts multiple threads which are tasked to find blocks in parallel, given some constraints.  This is achieved
        by calling "find_multiple_blocks" on each thread.

        :param n_blocks_per_thread: how many blocks each thread should try to produce.
        that were seen previously.
        :param n_threads: number of threads to use
        :return: list of Process objects
        """

        processes = []
        for thread_id in range(n_threads):
            p = Process(target=self.find_multiple_blocks,
                        args=(randint(0, 10000000), n_blocks_per_thread))
            processes.append(p)
            p.start()

        for proc in processes:
            proc.join()

        return processes

    def compute_term_idf_scores(self):
        """
        Computes IDF scores for each term (attribute value) in each view; very frequent terms have low IDF.
        """
        for view in self.tensor_stats_noidf:
            term_frequencies = self.tensor_stats_noidf[view][3].copy()
            term_idf_score_map = {}
            for term, term_freq in term_frequencies.items():
                term_idf_score_map[term] = np.log(1 + (1.0 / float(term_freq))) ** 2
            self.term_idf[view] = term_idf_score_map

    def cull_attr_to_entity_map(self):
        """
        Used to cull the attribute to entity mapping by excluding unshared attribute values, and values which have
        0 idf (cannot contribute mass to a block, and are thus useless).
        """

        for view in self.tensor_stats_noidf:
            culled_map = dict((key, value) for key, value in self.tensor_stats_noidf[view][4].items() if
                              len(value) > 1 and self.term_idf[view][key] > 0)
            self.tensor_stats_noidf[view] += (culled_map,)

    def __init__(self, entity_data, job_params):
        """
        Instantiates SliceNDice instance, given data and job parameters.

        :param entity_data: Pandas dataframe where rows are entities, and columns are attributes (pandas dataframe)
        :param job_params:  SliceNDiceJobParams instance containing job metadata
        """

        self.job_params = job_params
        logging.info('Initializing SliceNDice instance with job params {}'.format(job_params))

        # compute dictified version of all dataframe rows
        self.data_dict = entity_data.to_dict('index')
        self.all_entities = set(entity_data[self.job_params.ENTITY_ID_FIELD])

        # compute background metadata, without IDF
        self.tensor_stats_noidf = self.compute_block_metadata(self.all_entities, disable_idf=True,
                                                              return_attr_to_entity_map=True)

        self.term_idf = {}
        self.compute_term_idf_scores()

        # compute background metadata, with IDF
        self.tensor_stats = self.compute_block_metadata(self.all_entities, disable_idf=False)

        # background metadata of *remaining tensor* which respects local whitelisting.
        self.remaining_tensor_stats = deepcopy(self.tensor_stats)

        # compute culled attr_to_entity_maps for efficient seed selection
        self.cull_attr_to_entity_map()

        logging.info('Computed all background block metadata')
