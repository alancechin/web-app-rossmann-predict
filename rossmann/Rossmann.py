import pickle
import inflection
import pandas as pd
import numpy as np
import math
import datetime

## DATA CLEANING + FEATURE ENGINEERING + DATA FILTERING + DATA PREPARATION + FEATURE SELECTION

class Rossmann( object ):
    def __init__( self ):
        self.home_path = ''
        self.competition_distance_scaler   = pickle.load(open(self.home_path + 'parameter/competition_distance_scaler.pkl', 'rb'))
        self.competition_time_month_scaler = pickle.load(open(self.home_path + 'parameter/competition_time_month_scaler.pkl', 'rb'))
        self.promo_time_week_scaler        = pickle.load(open(self.home_path + 'parameter/promo_time_week_scaler.pkl', 'rb'))
        self.year_scaler                   = pickle.load(open(self.home_path + 'parameter/year_scaler.pkl', 'rb'))
        self.store_type_encoding           = pickle.load(open(self.home_path + 'parameter/store_type_encoding.pkl', 'rb'))


    def data_cleaning(self, df1):

        ## 1.1. Rename Columns ---------------------------------------------------------------------

        cols_old = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo',
                   'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
                   'CompetitionDistance', 'CompetitionOpenSinceMonth',
                   'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
                   'Promo2SinceYear', 'PromoInterval']

        ## Utiliza função lambda para encapsular a função que transforma para snake_case os atributos
        snakecase = lambda x: inflection.underscore( x )

        ## Map - Mapeia cada valor dentro da estrutura de dado passada e aplica a função snakecase
        cols_new = list( map( snakecase, cols_old ) )

        ## rename
        df1.columns = cols_new

        ## 1.3. Data Types ---------------------------------------------------------------------

        ## Arrumar para a coluna date ficar com o tipo de dado correto
        df1['date'] = pd.to_datetime( df1['date'] )


        ## 1.5. Fillout NA ---------------------------------------------------------------------

        #competition_distance

        # Assumptions: o valor NA ocorre quando a loja em questão não possui competidor próximo ou esse é tão longe que não é
        # considerado competidor próximo

        # Action: substituir os NA´s por um valor maior que o valor máximo existente para outras lojas

        df1['competition_distance'] = df1['competition_distance'].apply( lambda x: 200000.0 if math.isnan( x ) else x)

        #competition_open_since_month

        # Assumptions: se a loja não possui competidor próximo, até faz sentido essa coluna ser 0 ou NA, porque não existe data de
        # abertura se não existe loja, mas a questão é que existe mais faltante nessa coluna que na coluna de distância de competidor
        # próximo.

        # Action: substituir os NA´s pelo valor do mês da data de venda do histórico de vendas da loja. (sem muita lógica)

        df1['competition_open_since_month'] = df1.apply(lambda x: x['date'].month if math.isnan( x['competition_open_since_month'] )
                                                        else x['competition_open_since_month'], axis = 1)

        #competition_open_since_year

        # Assumptions: se a loja não possui competidor próximo, até faz sentido essa coluna ser 0 ou NA, porque não existe data de
        # abertura se não existe loja, mas a questão é que existe mais faltante nessa coluna que na coluna de distância de competidor
        # próximo.

        # Action: substituir os NA´s pelo valor do ano da data de venda do histórico de vendas da loja. (sem muita lógica)


        df1['competition_open_since_year'] = df1.apply(lambda x: x['date'].year if math.isnan( x['competition_open_since_year'] )
                                                        else x['competition_open_since_year'], axis = 1)

        #promo2_since_week

        # Assumptions: os NA´s ocorrem em lojas que não participam da promo2, pois não existe semana de início da promo2 se a loja
        # decidiu não participar.

        # Action: substituir os NA´s pelo valor da semana da data de venda do histórico de vendas da loja. (sem muita lógica)

        df1['promo2_since_week'] = df1.apply(lambda x: x['date'].week if math.isnan( x['promo2_since_week'] )
                                                        else x['promo2_since_week'], axis = 1)

        #promo2_since_year

        # Assumptions: os NA´s ocorrem em lojas que não participam da promo2, pois não existe ano de início da promo2 se a loja
        # decidiu não participar.

        # Action: substituir os NA´s pelo valor do ano da data de venda do histórico de vendas da loja. (sem muita lógica)

        df1['promo2_since_year'] = df1.apply(lambda x: x['date'].year if math.isnan( x['promo2_since_year'] )
                                                        else x['promo2_since_year'], axis = 1)

        #promo_interval

        # Assumptions: épocas de promoção 2 são conjuntos de meses que caracterizam o início de promoções 2. O NA nessa coluna
        # caracteriza que a loja referida não realizou promoções em nenhum mês do ano da referente data.

        # Action: substituir os NA´s por 0 pois significa que não tem época de promoção naquele ano.
        # Criar coluna extra ('is_promo') para indicar se mês de venda da loja vigente é o mês de promo2 consecutiva.

        month_map = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}

        df1['promo_interval'].fillna(0, inplace = True)

        #month_map é a abreviação do mês de mensuração de venda
        df1['month_map'] = df1['date'].dt.month.map( month_map )

        # atributo que diz se o mês de venda é o mês em que se realiza promoção periodicamente
        df1['is_promo'] = df1[['month_map','promo_interval']].apply(lambda x: 0 if x['promo_interval'] == 0 else 1 if
                                                                    x['month_map'] in x['promo_interval'].split( ',' ) else 0, axis = 1)


        ## 1.6. Change Types ---------------------------------------------------------------------


        ### Mudar nº de mês, ano e semana de float64 para int64

        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype( 'int64' )
        df1['competition_open_since_year'] = df1['competition_open_since_year'].astype( 'int64' )

        df1['promo2_since_week'] = df1['promo2_since_week'].astype( 'int64' )
        df1['promo2_since_year'] = df1['promo2_since_year'].astype( 'int64' )

        return df1


    def feature_engineering(self, df2):

        ## FEATURE ENGINEERING + DATA FILTERING

        ## 2.4. Feature Engineering

        ### 2.4.1. Da variável 'date'

        # year - extrair apenas o ano de vendas daquela loja

        df2['year'] = df2['date'].dt.year

        # month - extrair apenas o mês de vendas daquela loja

        df2['month'] = df2['date'].dt.month

        # day - extrair apenas o dia de vendas daquela loja

        df2['day'] = df2['date'].dt.day

        # week of year - extrair apenas a semana de vendas do ano daquela loja

        df2['week_of_year'] = df2['date'].dt.isocalendar().week
        # (year-week) - Formatar a data para em string aparecer o ano e a semana do ano das vendas daquela loja

        df2['year_week'] = df2['date'].dt.strftime("%Y-%W")


        ### 2.4.2. Das variáveis 'competition_open_since_month' e 'competition_open_since_year'

        # competition since
        df2['competition_since'] = df2[['competition_open_since_year','competition_open_since_month']].apply(lambda x: datetime.datetime(year = x['competition_open_since_year'] , month = x['competition_open_since_month'] , day = 1 ), axis = 1)

        # competition time month - Calculo de período entre datas
        df2['competition_time_month'] =( (df2['date'] - df2['competition_since']) / 30 ).apply( lambda x: x.days ).astype( 'int64' )

        ### 2.4.3. Das variáveis ''promo2_since_week'' e 'promo2_since_year'

        # promo since
        df2['promo_since'] =  df2['promo2_since_year'].astype( str ) + "-" + df2['promo2_since_week'].astype( str )

        # datetime
        df2['promo_since'] = df2['promo_since'].apply( lambda x: datetime.datetime.strptime( x + '-1', '%Y-%W-%w') - datetime.timedelta(days = 7) )


        # promo time week - - Calculo de período entre datas
        df2['promo_time_week'] =( (df2['date'] - df2['promo_since']) / 7 ).apply(lambda x: x.days ).astype( 'int64' )

        ### 2.4.4. Da variável 'assortment'
        df2['assortment'] = df2['assortment'].apply(lambda x: 'basic' if x == 'a' else 'extra' if x == 'b' else 'extended')

        ### 2.4.5. Da variável 'state_holiday'
        df2['state_holiday'] = df2['state_holiday'].apply(lambda x: 'public_holiday' if x == 'a' else 'easter_holiday' if x == 'b' else 'christmas' if x == 'c' else 'regular_day')

        # 3.0. FILTRAGEM DE VARIÁVEIS

        ## 3.1. Filtragem das Linhas

        df2 = df2[ df2['open'] != 0  ]

        ## 3.2. Seleção das Colunas

        ## COLUNAS EXCLUÍDAS:
        cols_drop = ['open','promo_interval','month_map']
        df2 = df2.drop( cols_drop, axis = 1 )

        return df2

    def data_preparation(self, df5):


        ## 5.2. Rescaling

        # competition_distance
        df5["competition_distance"] = self.competition_distance_scaler.transform( df5[["competition_distance"]].values )

        # competition_time_month
        df5["competition_time_month"] = self.competition_time_month_scaler.transform( df5[["competition_time_month"]].values )

        # promo_time_week
        df5["promo_time_week"] = self.promo_time_week_scaler.transform( df5[["promo_time_week"]].values )

        # year
        df5["year"] = self.year_scaler.transform( df5[["year"]].values )

        ## 5.3. Transformação

        ### 5.3.1. Encoding

        # state_holiday -> One Hot Encoding
        df5 = pd.get_dummies(df5, prefix = ['state_holiday'], columns = ['state_holiday'])

        # store_type - Label Encoder
        df5["store_type"] = self.store_type_encoding.transform( df5["store_type"] )

        # assortment
        assortment_dict = {'basic': 1, 'extended': 2, 'extra': 3}
        # Troca de valores da coluna utilizando o método .map e o dicionário como referência
        df5["assortment"] = df5["assortment"].map(assortment_dict)


        ### 5.3.3. Nature Transformation

        ## FEATURES COM NATUREZA CÍCLICA:

        # day_of_week - ciclo de 7
        df5["day_of_week_sin"] = df5["day_of_week"].apply(lambda x: np.sin( x * ( 2. * np.pi/7 ) ) )
        df5["day_of_week_cos"] = df5["day_of_week"].apply(lambda x: np.cos( x * ( 2. * np.pi/7 ) ) )


        # month - ciclo de 12
        df5["month_sin"] = df5["month"].apply(lambda x: np.sin( x * ( 2. * np.pi/12 ) ) )
        df5["month_cos"] = df5["month"].apply(lambda x: np.cos( x * ( 2. * np.pi/12 ) ) )


        # day - ciclo de 31
        df5["day_sin"] = df5["day"].apply(lambda x: np.sin( x * ( 2. * np.pi/31 ) ) )
        df5["day_cos"] = df5["day"].apply(lambda x: np.cos( x * ( 2. * np.pi/31 ) ) )


        # week_of_year - ciclo de 52
        df5["week_of_year_sin"] = df5["week_of_year"].apply(lambda x: np.sin( x * ( 2. * np.pi/52 ) ) )
        df5["week_of_year_cos"] = df5["week_of_year"].apply(lambda x: np.cos( x * ( 2. * np.pi/52 ) ) )

        # Feature Selection with Boruta and EDA
        cols_selected = ['store', 'promo', 'store_type','assortment', 'competition_distance', 'competition_open_since_month',
        'competition_open_since_year', 'promo2', 'promo2_since_week','promo2_since_year', 'competition_time_month',
        'promo_time_week','day_of_week_sin','day_of_week_cos','month_cos','month_sin','day_sin','day_cos','week_of_year_cos',
        'week_of_year_sin']


        return df5[cols_selected]


    def get_prediction( self, model, original_data, test_data ):
        # prediction
        pred = model.predict( test_data )

        # join pred into the original data
        original_data['prediction'] = np.expm1( pred )

        return original_data.to_json( orient = 'records', date_format = 'iso' )
