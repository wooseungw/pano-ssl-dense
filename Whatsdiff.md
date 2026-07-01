1. img-text grounging 구조를 가지지 않아도 됨
    이에따라 아주 독특한 구조를 사용 가능함
2. vision task 최적화 가능
    주요 대상 segmentation, 3D Detection, depth estimation
    pointmap 생성
        포인트맵 생성은 기존의 colmap(하나의 대상을 여러 각도로 촬영)과 달리 하나의 지점에서 장면을 획득
        이거는 좀 확인해야할듯 성능을
            pinhole변환시킨걸 오버랩 있게 이어붙힌것과
3. ssl 기법으로 인해 cubemap,anyres-e2p, 등과 같은 파노라마 이미지를 처리하기위한 기법과 모두 호환가능
4. 월드모델 지향
    얇은 디코더층을 대상으로한 다운스트림 작업
