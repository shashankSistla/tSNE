from IPython.display import display, clear_output, HTML
import os
import ipywidgets as widgets 
from src.base import is_previously_computed, get_tsne_data
import pandas as pd
import json

from bokeh.plotting import figure, show
from bokeh.models import HOverTool, LinearColorMapper, ColorBar
from bokeh.models import ColumnDataSOurce, HoverTool, CustomJS
from boeh.transform import factor_cmap, linear_cmap
from bokeh.io import output_notebook
from bokeh.palettes import Category10, Category20
from bokeh.layouts import row

from scipy.stats import ttest_ind, mannwhitneyu

import pickle

def custom_formatter_numeric(x):
    try:
        return "{:. .3g}".format(x)
    except:
        return x
    
def get_hover_columns(key):
    conf_dir = os.getcwd() + f'/keys/{key}/conf.json'
    f = open(conf_dir)
    conf = json.load(f)
    return conf['hover_columns']


def get_exclude_columns(key):
    conf_dir = os.getcwd() + f'/keys/{key}/conf.json'
    f = open(conf_dir)
    conf = json.load(f)
    return conf['exclude_columns']

def get_stat_test_columns(key):
    conf_dir = os.getcwd() + f'/keys/{key}/conf.json'
    f = open(conf_dir)
    conf = json.load(f)
    return conf['stat_test_columns']

def get_input_file(key):
    conf_dir = os.getcwd() + f'/keys/{key}/conf.json'
    f = open(conf_dir)
    conf = json.load(f)
    return conf['input_file']

def get_default_groupby(key):
    conf_dir = os.getcwd() + f'/keys/{key}/conf.json'
    f = open(conf_dir)
    conf = json.load(f)
    return conf['default_groupby']

def get_key_list():
    keys_dir = './keys'
    if not os.path.exists(keys_dir):
        os.makedirs(keys_dir)

    key_names = [d for d in os.listdir(keys_dir) if os.path.isdir(os.path.join(keys_dir, d))]
    return key_names

def get_perp_list(key):
    perp_dir = f'./keys/{key}/tSNE_data'
    if not os.path.exists(perp_dir):
        os.makedirs(perp_dir)

    perp_list = [(d.split('_')[1]).split('.')[0] for d in os.listdir(perp_dir) if (os.path.isfile(os.path.join(perp_dir, d))) and d[0:4] == 'tsne']

    return perp_list

def lerp_color(color1, color2, factor):
    return tuple(int(c1 + (c2 - c1) * factor) for c1, c2 in zip(color1, color2))

def create_gradient(start_color, end_color, n = 256):
    return [lerp_color(start_color, end_color, i/(n-1)) for i in range(n)]

def get_gradient_hex():
    start_color = (255, 0, 0)
    middle_color = (255, 255, 255)
    end_color = (0,0, 255)

    red_to_white = create_gradient(start_color, middle_color, n = 128)
    white_to_blue = create_gradient(middle_color, end_color, n = 128)

    full_gradient = red_to_white + white_to_blue[1:]

    full_gradient_hex = ['#%02x%02x%02x' % color for color in full_gradient]

    return full_gradient_hex


class SelectorInterface:

    def __init__(self):
        self.df = None
        self.inds = []
        self.log = ""
        self.key_list = get_key_list()
        self.key = self.key_list[0]
        self.filters_df = pd.DataFrame(columns = ['column', 'Type', 'Values'])
        self.data_df = pd.read_csv(get_input_file(), low_memory=False)
        self.is_data_loaded = False
        self.groupby_column = None
        self.groupby_column_unique_members = None
        self.perp_list = get_perp_list(self.key)

        # Inputs
        self.key_dropdown = widgets.Dropdown(options = self.key_list, description = 'Select Key:', value = self.key_list[0])
        self.perplexity_dropdown = widgets.Dropdown(options = self.perp_list, description = 'Select Perplexity:', style={'description_width': 'initial'}, value = self.perp_list[0])
        self.file_input = widgets.Text(value= '', placeholder = 'Enter Cluster Name', description = 'Cluster Name:')

        # Buttons
        self.save_filters_to_key_button = widgets.Button(description = 'Save_Filters to Key')
        self.apply_filters_button = widgets.Button(description = 'Apply Filters')
        self.create_filter_button = widgets.Button(description = 'Create Filter')
        self.save_button = widgets.Button(description = 'Save Selected Points')
        self.compute_analysis_button = widgets.Button(description = "Compute Analysis")

        # Outputs

        self.message_output = widgets.Output()
        self.filtered_data_output = widgets.Output()
        self.bokeh_plot_output = widgets.Output()
        self.error_message_output = widgets.Output()

        self.create_filter_output = widgets.Output()
        self.filter_view_output = widgets.Output()

        # Attacing actions to inputs

        self.key_dropdown.observe(self.on_key_dropdown_change)
        self.perplexity_dropdown.observe(self.on_perplexity_change)
        self.save_filters_to_key_button.on_click(self.save_filters_to_key)
        self.save_button.on_click(self.on_save_button_clicked)
        self.compute_analysis_button.on_click(self.on_compute_analysis_button_clicked)
        self.apply_filters_button.on_click(self.apply_filters)
        self.create_filter_button.on_click(self.show_create_filter_ui)

    def interface(self): #Entry point
        self.display_widgets_interface()
        self.load_key()
        path = f'./keys/{self.key}/tSNE_data'
        self.show_bokeh_plot(path, self.perp_list[0])

    def display_widgets_interface(self):
        display(widgets.VBox([
            widgets.HBox([self.key_dropdown, self.create_filter_button, self.save_filters_to_key_button]),
            widgets.VBox([self.filter_view_output, self.create_filter_output, self.apply_filters_button, self.filtered_data_output])
        ]))

        display(self.perplexity_dropdown, self.compute_analysis_button, self.bokeh_plot_output, self.file_input, self.save_button, self.error_message_output)


    # Actions = method which the inputs interact with
    def on_key_dropdown_change(self, change):
        if change['type'] == 'change' and change['name'] == 'value':
            self.load_key()

    def load_key(self):
        self.key = self.key_dropdown.value
        filters_file_path = f'./keys/{self.key}/filters.json'
        self.groupby_column = get_default_groupby(self.key)
        self.perp_list = get_perp_list(self.key)

        self.perplexity_dropdown.options = self.perp_list
        self.perplexity_dropdown.value = self.perp_list[0]

        if os.path.exists(filters_file_path):
            with open(filters_file_path, 'r') as f:
                self.filters_df = pd.DataFrame(json.load(f))

            self.update_filter_view()
            self.display_message(f'Loaded filters from key: {self.key}')
        else:
            self.filters_df = pd.DataFrame(columns = ['Column', 'Type', 'Values']) #Reset filters_df
            self.update_filter_view()
            self.display_message(f'No filters found for key: {self.key}. Add filters if necessary')

    def on_perplexity_change(self, b):
        if b['name'] == 'value':
            perplexity = self.perplexity_dropdown.value
            path = f'./keys/{self.key}/tSNE_data'
            self.show_bokeh_plot(path, perplexity)


    def save_filters_to_key(self, b):
        filters_file_path = f'./keys/{self.key}/filters.json'

        with open(filters_file_path, 'w') as f:
            json.dump(self.filters_df.to_dict('records'), f, indent = 4)
        self.display_message(f'Saved filters to key: {self.key}')

    
    def show_bokeh_plot(self, path, perplexity):
        self.df = get_tsne_data(path, perplexity)
        self.clear_bokeh_plot()

        with self.bokeh_plot_output:
            self.select_indices()

    def on_save_button_clicked(self, event):
        with self.error_message_output:
            filename = self.file_input.value if not filename:
            print("Error: Please provide a filename")

        filepath = f"./keys/{self.key}/stored_indices/{filename}.pkl"

        if os.path.exists(filepath):
            print(f"Error: File '{filename}.pkl already exists. Choose a differet name.")
            return
        
        if not self.inds:
            print("Error: No points selected")
            return
        pickle.dump(self.inds, open(filepath, 'wb'))
        print(f"Succesfully saved to '{filename}.pkl!")

    def on_compute_analysis_button_clicked(self, event):
        if self.inds:
            result_df = self.difference_analysis()
            display(result_df.style.format(custom_formatter_numeric))
        else:
            print("No points selected.")

    def apply_filters(self, b):
        filtered_df = self.data_df.copy()
        for index, filter_row in self.filters_df.iterrows():
            column = filter_row['Column']
            filter_type = filter_row['Type']
            values = filter_row['Values']
            if filter_type == 'number':
                lower_bound, upper_bound = map(float, values.split(' to '))
                filtered_df = filtered_df[(filtered_df[column].astype(float) >= lower_bound) & (filtered_df[column].astype(float) <= upper_bound)] 

            elif filter_type == 'category':
                selected_categories = values.split(', ')
                filtered_df = filtered_df[filtered_df[column].isin(selected_categories)]

        with self.filtered_data_output:
            clear_output()
            display(filtered_df)


    def clear_bokeh_plot(self):
        self.create_filter_button.disabled = True
        with self.bokeh_plot_output:
            clear_output()
            create_filter_widgets()

        def create_filter_widgets():
            column_dropdown = widgets.Dropdown(options = self.data.df.columns.tolist(), description = 'Column')
            parse_as_dropdown = widgets.Dropdown(options = ['number', 'category'], description= 'Parse As: ')
            
            lower_bound = widgets.FloatText(value = 0, description = 'Lower Bound:' )
            upper_bound = widgets.FloatText(value = 0, descirption = 'Upper Bound:' )

            category_select = widgets.SelectMultiple(options = [], description = 'Categories:')
            add_button = widgets.Button(description = 'Add')
            cancel_button = widgets.Button(description = 'Cancel')

            def update_ui(change):
                if parse_as_dropdown.value == 'number':
                    ui.children = [column_dropdown, parse_as_dropdown, lower_bound, upper_bound, buttons]
                else:
                    unique_values = self.data_df[column_dropdown.value].unique().tolist()
                    category_select.options = unique_values
                    ui.children = [column_dropdown, parse_as_dropdown, category_select, buttons]

            parse_as_dropdown.observe(update_ui, 'value')

            def add_filter(b):
                if parse_as_dropdown.value == 'number':
                    values = f'{lower_bound.value} to {upper_bound.value}'
                else:
                    value = ', '.join(map(str, category_select.value))
                
                self.filters_df = self.filters_df.append({'Column': column_dropdown.value, 'Type': parse_as_dropdown.value, 'Values': values}, ignore_index = True)


                self.update_filter_view()
                with self.create_filter_output:
                    clear_output()

                self.create_filter_button.disabled = False

            def cancel_filter(b):
                    with self.create_filter_output:
                        clear_output()
                    self.create_filter_button.disabled = False

            add_button.on_click(add_filter)
            cancel_button.on_click(cancel_filter)

            buttons = widgets.HBox([add_button, cancel_button])
            ui = widgets.VBox([column_dropwdown, parse_as_dropdown, buttons])

            display(ui)


        def update_filter_view(self):
            with self.filter_view_output:
                clear_output()
                rows = []

                for index, row in self.filters_df.iterrows():
                    delete_button = widgets.Button(description = 'Delete')
                    delete_button.on_click(lambda b, index=index: self.delete_filter(index))
                    filter_description = widgets.Label(value = f'{row.column} ({row.Type}): {row.Values}')
                    row_widgets = widgets.HBox([filter_description, delete_button], layout = widgets.Layout(justify_content = 'space-between', border = '1px solid #ddd', padding='8px'))

                    rows.append(row_widgets)
                display(widgets.VBox(rows))

        def delete_filter(self, index):
            self.filters_df = self.filters_df.drop(index)
            self.update_filter_view()


        def display_message(self, message):
            with self.message_output:
                clear_output(wait = True)
                print(message)

        def display_difference_analysis(self):
            if self.inds:
                result_df = self.difference_analysis()
                clear_output(wait = True)
                display(result_df)

        def select_indices(self, s = 3):
            unique_column_members = self.df[self.groupby_column].unique()

            x_values = self.df['X']
            y_values = self.df['Y']

            hover_data = {}
            hover_columns = get_hover_columns(self.key)

            for col in hover_columns:
                hover_data[col] = self.df[col]

            s1 = ColumnDataSource(data = dict(x=x_values, y=y_values, **hover_data))
            tools = "pan, wheel_zoom, reset, lasso_select, box_select, tap"

            p1 = figure(width = 600, height = 600, tools = tools, title = 'Select Here')

            if len(unique_column_members) <= 20:
                palette = Category10[10] if len(unique_column_members) <= 10 else Category20[20]
                index_cmap = factor_cmap(self.groupby_column, palette = palette, factors = unique_column_members)
                p1.scatter('x', 'y', source = s1, alpha = 0.6, size = s, fill_color = index_cmap, line_width = 0 )

            else:
                full_gradient_hex = get_gradient_hex()
                mapper = LinearColorMapper(palette = full_gradient_hex, low = min(self.df[self.groupby_column]), high = max(self.df[self.groupby_column]))
                color_bar = ColorBar(color_apper = mapper, width = 8, location = (0,0))
                p1.scatter('x','y', source=s1, alpha=0.6, size = s, color = linear_cmap(self.groupby_column, full_gradient_hex, min(self.df[self.groupby_column]), max(self.df[self.groupby_column])), line_width = 0)
                p1.add_layout(color_bar, 'right')

            hover = HoverTool()
            hover.tooltips = [(col, f'@{col}') for col in hover_columns]
            p1.add_tools(hover)

            s1.selected.js_on_change('indices', CustomJS(code = """
            const inds = cb_obj.indices;
            console.log("Indices changed");
            IPython.notebook.kernel.execute("print('Test')");
            IPython.notebook.kernel.execute("interface.inds = " + inds);                                                         
            IPython.notebook.kernel.execute("interface.log = Help done");
            """))

            output_notebook()
            show(p1, notebook_handle = True)

        def set_inds(self, inds):
            self.inds = inds

        def difference_analysis(self):
            exclude_columns = get_exclude_columns(self.key)

            exclude_columns.append('X')
            exclude_columns.append('Y')

            cols = [x for x in self.df.columns if x not in exclude_columns]

            group_interest = self.df.iloc[list(self.inds)].drop(columns = exclude_columns)
            group_others = self.df.drop(list(self.inds)).drop(columns = exclude_columns)

            median_interest = group_interest.median()
            median_others = group_others.median()

            variance_interest = group_interest.var()
            variance_others = group_others.var()

            p_vals = []
            for col in group_interest.columns:
                _, p = mannwhitneyu(group_interest[col], group_others[col], alternative = 'two-sided')
                p_vals.append(p)

            #t_stats, p_vals = ttest_ind(group_interst, group_others)

            results = pd.DataFrame({
                'Feature': [x for x in self.df.columns if x not in exclude_columns],
                'Group of Interest Median': median_interest,
                'Other Group Median': median_others,
                'P-Value': p_vals,
            }).sort_values(by='P-Value', ascending = True)

            p_vals = []
            stat_test_df = None
            if cols:
                for col in cols:
                    _, p = mannwhitneyu(group_interest[col], group_others[col], alternative='two-sided')
                    p_vals.append(p)
        
            top_5_features = results.head(10)
            return top_5_features.reset_index(drop = True)





            

    










