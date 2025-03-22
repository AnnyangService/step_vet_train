import numpy as np

class ChannelSelector:
    def __init__(self, padding_config=None, scoring_config=None):
        # 기본 설정값 정의
        self.padding_config = {
            'threshold': 0.8,    # 패딩 판단 임계값
            'height_ratio': 8    # 패딩 영역 비율 (height // ratio)
        } if padding_config is None else padding_config

        self.scoring_config = {
            'contrast_weight': 0.5,      # contrast score 가중치
            'edge_weight': 0.5,          # edge score 가중치
            'high_percentile': 90,       # contrast 상위 퍼센타일
            'low_percentile': 10,        # contrast 하위 퍼센타일
            'top_k': 30                  # 선택할 상위 채널 수
        } if scoring_config is None else scoring_config

    def check_padding_activation(self, channel):
        """패딩 영역 활성화 체크"""
        h = channel.shape[0]
        padding_height = h // self.padding_config['height_ratio']
        
        top_pad = channel[:padding_height].mean()
        bottom_pad = channel[-padding_height:].mean()
        center = channel[padding_height:-padding_height].mean()
        
        return (top_pad > center * self.padding_config['threshold']) or \
               (bottom_pad > center * self.padding_config['threshold'])

    def calculate_channel_scores(self, features):
        """채널 스코어 계산"""
        scores = []
        padding_excluded = []
        
        for i in range(features.shape[1]):
            channel = features[0, i].cpu().numpy()
            
            # 패딩 체크
            if self.check_padding_activation(channel):
                scores.append(-float('inf'))
                padding_excluded.append(i)
                continue
            
            # a. Contrast score
            high_thresh = np.percentile(channel, self.scoring_config['high_percentile'])
            low_thresh = np.percentile(channel, self.scoring_config['low_percentile'])
            contrast_score = high_thresh - low_thresh
            
            # b. Edge detection score
            grad_x = np.gradient(channel, axis=1)
            grad_y = np.gradient(channel, axis=0)
            edge_score = np.mean(np.sqrt(grad_x**2 + grad_y**2))    
            
            # 최종 점수 계산
            final_score = (
                contrast_score * self.scoring_config['contrast_weight'] +
                edge_score * self.scoring_config['edge_weight']
            )
            
            scores.append(final_score)
        
        # 상위 k개 채널 선택
        top_channels = np.argsort(scores)[-self.scoring_config['top_k']:][::-1]
        return top_channels, scores, padding_excluded

    def select_channels(self, features):
        """특징에서 중요 채널 선택"""
        top_channels, scores, padding_excluded = self.calculate_channel_scores(features)
        return features[:, top_channels], top_channels
        
    def get_channel_vector(self, features):
        """특징에서 중요 채널 선택 후 벡터화"""
        top_channels, _, _ = self.calculate_channel_scores(features)
        
        # 선택된 채널들의 특징맵을 numpy로 변환
        selected_features = np.stack([
            features[0, idx].cpu().numpy() 
            for idx in top_channels[:self.scoring_config['top_k']]
        ])
        
        # 채널 방향으로 평균 계산
        mean_feature_map = np.mean(selected_features, axis=0)
        
        # 특징맵을 1차원 벡터로 변환
        feature_vector = mean_feature_map.flatten()
        
        return feature_vector