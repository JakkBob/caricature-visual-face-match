'use client';

import { useState, useCallback } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Label } from '@/components/ui/label';
import { Separator } from '@/components/ui/separator';
import {
  ImageUpload,
  MatchResult,
  EvaluationPanel,
  SystemStatus,
  DatasetManager,
} from '@/components/face-match';
import { MatchedPair, EvaluationMetrics } from '@/types';
import {
  ArrowRightLeft,
  Loader2,
  Sparkles,
  BarChart3,
  Database,
  Settings,
  Github,
} from 'lucide-react';

type MatchMode = 'face-to-caricature' | 'caricature-to-face';

export default function Home() {
  // 匹配状态
  const [matchMode, setMatchMode] = useState<MatchMode>('caricature-to-face');
  const [queryImage, setQueryImage] = useState<string | null>(null);
  const [queryFile, setQueryFile] = useState<File | null>(null);
  const [matches, setMatches] = useState<MatchedPair[]>([]);
  const [isMatching, setIsMatching] = useState(false);
  const [processTime, setProcessTime] = useState<number | undefined>(undefined);

  // 评估状态
  const [metrics, setMetrics] = useState<EvaluationMetrics | undefined>();
  const [isEvaluating, setIsEvaluating] = useState(false);

  // 当前标签页
  const [activeTab, setActiveTab] = useState('match');

  const handleImageSelect = useCallback((file: File, preview: string) => {
    setQueryFile(file);
    setQueryImage(preview);
    setMatches([]);
    setProcessTime(undefined);
  }, []);

  const handleImageClear = useCallback(() => {
    setQueryFile(null);
    setQueryImage(null);
    setMatches([]);
    setProcessTime(undefined);
  }, []);

  const handleMatch = async () => {
    if (!queryImage) return;

    setIsMatching(true);
    try {
      // 将Base64图像数据转换为纯Base64字符串
      const base64Data = queryImage.includes(',')
        ? queryImage.split(',')[1]
        : queryImage;

      const response = await fetch('/api/match', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          queryImage: base64Data,
          queryModality: matchMode === 'face-to-caricature' ? 'face' : 'caricature',
          targetModality: matchMode === 'face-to-caricature' ? 'caricature' : 'face',
          topK: 10,
        }),
      });

      const data = await response.json();
      console.log('[Match] Response data:', JSON.stringify(data, null, 2));
      
      if (data.success && data.data) {
        // Transform backend response to MatchedPair format
        // Handle nested data structure: data.data.matches or data.matches
        const matchesData = data.data.matches || data.matches || [];
        console.log('[Match] Matches data:', JSON.stringify(matchesData, null, 2));
        
        const transformedMatches: MatchedPair[] = matchesData.map(
          (match: { index?: number; similarity: number; rank: number; isMatch?: boolean; is_match?: boolean; id?: string; image_data?: string }) => {
            // Ensure similarity is a valid number
            const similarity = typeof match.similarity === 'number' ? match.similarity : 0;
            console.log('[Match] Processing match:', { id: match.id, similarity, rank: match.rank });
            return {
              imageId: match.id || `match-${match.index || match.rank}`,
              imagePath: '', // Backend doesn't return path, will show placeholder
              imageData: match.image_data, // Base64 image data from backend
              similarity: similarity,
              rank: match.rank || 1,
              isMatch: match.isMatch ?? match.is_match ?? false,
            };
          }
        );
        console.log('[Match] Transformed matches:', JSON.stringify(transformedMatches, null, 2));
        setMatches(transformedMatches);
        
        // Get process time from response (try multiple locations)
        const processTimeMs = data.data.process_time || data.data.processTime || data.process_time || data.processTime || 0;
        console.log('[Match] Process time:', processTimeMs);
        setProcessTime(processTimeMs);
      } else {
        console.error('Match failed:', data.message);
      }
    } catch (error) {
      console.error('Match error:', error);
    } finally {
      setIsMatching(false);
    }
  };

  const handleEvaluate = async () => {
    setIsEvaluating(true);
    try {
      const response = await fetch('/api/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          saveResults: true,
        }),
      });

      const data = await response.json();
      if (data.success && data.data) {
        setMetrics(data.data.metrics);
      } else {
        console.error('Evaluation failed:', data.message);
      }
    } catch (error) {
      console.error('Evaluation error:', error);
    } finally {
      setIsEvaluating(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted/30">
      {/* 头部 */}
      <header className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-primary/10">
                <ArrowRightLeft className="h-6 w-6 text-primary" />
              </div>
              <div>
                <h1 className="text-xl font-bold">跨模态人脸识别系统</h1>
                <p className="text-sm text-muted-foreground">
                  漫画-视觉人脸匹配
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="hidden sm:flex">
                v1.0.0
              </Badge>
              <Button variant="ghost" size="icon" asChild>
                <a
                  href="https://github.com/JakkBob/caricature-visual-face-match"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Github className="h-5 w-5" />
                </a>
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* 主内容 */}
      <main className="container mx-auto px-4 py-6">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-3 lg:w-[400px] mx-auto">
            <TabsTrigger value="match" className="gap-2">
              <Sparkles className="h-4 w-4" />
              <span className="hidden sm:inline">匹配识别</span>
              <span className="sm:hidden">匹配</span>
            </TabsTrigger>
            <TabsTrigger value="evaluate" className="gap-2">
              <BarChart3 className="h-4 w-4" />
              <span className="hidden sm:inline">实验评估</span>
              <span className="sm:hidden">评估</span>
            </TabsTrigger>
            <TabsTrigger value="dataset" className="gap-2">
              <Database className="h-4 w-4" />
              <span className="hidden sm:inline">数据管理</span>
              <span className="sm:hidden">数据</span>
            </TabsTrigger>
          </TabsList>

          {/* 匹配识别标签页 */}
          <TabsContent value="match" className="space-y-6">
            <div className="grid lg:grid-cols-2 gap-6">
              {/* 左侧：上传和设置 */}
              <div className="space-y-4 lg:sticky lg:top-6 lg:self-start">
                {/* 匹配模式选择 */}
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-base">匹配模式</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <RadioGroup
                      value={matchMode}
                      onValueChange={(v) => setMatchMode(v as MatchMode)}
                      className="flex flex-col sm:flex-row gap-4"
                    >
                      <div className="flex items-center space-x-2 flex-1 p-3 rounded-lg border hover:bg-muted/50 transition-colors cursor-pointer">
                        <RadioGroupItem value="caricature-to-face" id="c2f" />
                        <Label htmlFor="c2f" className="cursor-pointer flex-1">
                          <span className="font-medium">漫画 → 真实人脸</span>
                          <p className="text-xs text-muted-foreground">
                            输入漫画图像，检索匹配的真实人脸
                          </p>
                        </Label>
                      </div>
                      <div className="flex items-center space-x-2 flex-1 p-3 rounded-lg border hover:bg-muted/50 transition-colors cursor-pointer">
                        <RadioGroupItem value="face-to-caricature" id="f2c" />
                        <Label htmlFor="f2c" className="cursor-pointer flex-1">
                          <span className="font-medium">真实人脸 → 漫画</span>
                          <p className="text-xs text-muted-foreground">
                            输入真实人脸图像，检索匹配的漫画
                          </p>
                        </Label>
                      </div>
                    </RadioGroup>
                  </CardContent>
                </Card>

                {/* 图像上传 */}
                <ImageUpload
                  modality={matchMode === 'face-to-caricature' ? 'face' : 'caricature'}
                  onImageSelect={handleImageSelect}
                  onImageClear={handleImageClear}
                  preview={queryImage || undefined}
                  disabled={isMatching}
                />

                {/* 匹配按钮 */}
                <Button
                  className="w-full h-12 text-base"
                  onClick={handleMatch}
                  disabled={!queryImage || isMatching}
                >
                  {isMatching ? (
                    <>
                      <Loader2 className="h-5 w-5 animate-spin mr-2" />
                      匹配中...
                    </>
                  ) : (
                    <>
                      <Sparkles className="h-5 w-5 mr-2" />
                      开始匹配
                    </>
                  )}
                </Button>
              </div>

              {/* 右侧：结果和状态 */}
              <div className="space-y-4">
                <SystemStatus />
                
                {matches.length > 0 ? (
                  <MatchResult
                    matches={matches}
                    queryModality={matchMode === 'face-to-caricature' ? 'face' : 'caricature'}
                    processTime={processTime}
                  />
                ) : (
                  <Card className="w-full">
                    <CardContent className="flex flex-col items-center justify-center py-12">
                      <p className="text-muted-foreground">上传图像后点击"开始匹配"查看结果</p>
                    </CardContent>
                  </Card>
                )}
              </div>
            </div>
          </TabsContent>

          {/* 实验评估标签页 */}
          <TabsContent value="evaluate" className="space-y-6">
            <div className="grid lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <EvaluationPanel
                  onEvaluate={handleEvaluate}
                  metrics={metrics}
                  isEvaluating={isEvaluating}
                />
              </div>
              <div>
                <SystemStatus />
              </div>
            </div>
          </TabsContent>

          {/* 数据管理标签页 */}
          <TabsContent value="dataset" className="space-y-6">
            <div className="grid lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <DatasetManager onRefresh={() => {}} />
              </div>
              <div>
                <SystemStatus />
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </main>

      {/* 页脚 */}
      <footer className="border-t mt-auto">
        <div className="container mx-auto px-4 py-4">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-2 text-sm text-muted-foreground">
            <p>跨模态漫画-视觉人脸识别系统 - 毕业论文项目</p>
            <p>© 2024 硕士研究生研究项目</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
